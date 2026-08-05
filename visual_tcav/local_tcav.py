"""
local_tcav.py
-------------
LocalVisualTCAV: explains a single test image using Visual-TCAV.

For each (concept, layer) pair, produces:
- A concept map: heatmap showing WHERE the CNN detected the concept
- Attribution scores: HOW MUCH the concept influenced each predicted class
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from visual_tcav.visual_tcav import VisualTCAV
from visual_tcav.utils import DEFAULT_COLORMAP

sys.dont_write_bytecode = True


class LocalVisualTCAV(VisualTCAV):
    """
    Explains a single test image using Visual-TCAV.

    The model is provided via one of two standard interfaces:

    **String** — auto-loads the model with default ImageNet weights:

    .. code-block:: python

        tcav = LocalVisualTCAV(
            model="resnet50",
            test_image_path="./zebra.jpg",
            concept_names=["striped", "dotted"],
            concept_base_dir="./concept_images",
            layer_names=["layer4"],
        )
        tcav.explain()
        tcav.plot()

    **nn.Module** — use your own model:

    .. code-block:: python

        import torchvision.models as models
        resnet = models.resnet50(weights='DEFAULT')
        tcav = LocalVisualTCAV(
            model=resnet,
            model_name="resnet50",
            ...
        )

    For advanced use cases (custom labels, custom preprocessing), use
    :class:`~visual_tcav.model_wrapper.TorchModelWrapper` directly and
    pass it via ``model_wrapper``.

    All parameters are optional at construction time. Validation happens
    when explain() is called, so you can create the object first and
    inspect available layers with model_wrapper.info() before configuring.

    Parameters
    ----------
    model : str or nn.Module, optional
        Model name string or PyTorch model object.
    model_name : str, optional
        Display name for the model. For nn.Module of known torchvision
        models, set this to enable auto-loading of labels
        (e.g. model_name="resnet50").
    model_wrapper : TorchModelWrapper, optional
        Pre-built wrapper for advanced use cases.
    test_image_path : str, optional
        Path to the image to explain.
    concept_names : list of str, optional
        Names of the concepts to analyze.
    concept_base_dir : str, optional
        Folder containing one subfolder per concept.
    concept_dirs : dict, optional
        Explicit mapping of concept name to image folder.
    random_dir : str, optional
        Folder containing random images used as reference distribution.
        Defaults to concept_base_dir/random/ if not provided.
    layer_names : list of str, optional
        CNN layers to analyze. Call model_wrapper.info() to see
        available layer names.
    n_classes : int, optional
        Number of top predicted classes to explain. Default is 3.
    m_steps : int, optional
        Interpolation steps for Integrated Gradients. Default is 50.
    max_examples : int, optional
        Maximum number of concept/random images. Default is 500.
    cache_dir : str, optional
        Directory for caching results. Default is ".cache".
    cav_fn : callable, optional
        Custom CAV computation function. Must accept two tensors of
        shape [N, C] and return a Cav object.
    """

    def __init__(
        self,
        model=None,
        model_name: str = None,
        model_wrapper=None,
        test_image_path: str = None,
        concept_names: list = None,
        concept_base_dir: str = None,
        concept_dirs: dict = None,
        random_dir: str = None,
        layer_names: list = None,
        n_classes: int = 3,
        m_steps: int = 50,
        max_examples: int = 500,
        cache_dir: str = ".cache",
        cav_fn=None,
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            model_wrapper=model_wrapper,
            n_classes=n_classes,
            m_steps=m_steps,
            max_examples=max_examples,
            cache_dir=cache_dir,
            cav_fn=cav_fn,
        )

        self.test_image_path = None
        self.test_image_tensor = None
        self.test_image_display = None

        if test_image_path is not None:
            self._load_test_image(test_image_path)

        if concept_names is not None:
            self._setup_concepts(
                concept_names=concept_names,
                concept_dirs=concept_dirs,
                concept_base_dir=concept_base_dir,
                random_dir=random_dir,
            )

        if layer_names is not None:
            self._setup_layers(layer_names)

    # -----------------------------------------------------------------------
    # Image loading
    # -----------------------------------------------------------------------

    def _load_test_image(self, image_path: str) -> None:
        """
        Load and preprocess the test image.

        Creates two versions:
        - A normalized tensor for model inference
        - An unnormalized PIL image for visualization

        Parameters
        ----------
        image_path : str
            Path to the image file (JPEG, PNG, or other PIL-supported formats).

        Raises
        ------
        FileNotFoundError
            If the image file does not exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Test image not found at: {image_path}")

        self.test_image_path = image_path
        image = Image.open(image_path).convert("RGB")
        C, H, W = self.model_wrapper.input_size

        preprocess = transforms.Compose([
            transforms.Resize(
                (H, W),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop((H, W)),
            transforms.ToTensor(),
            self.model_wrapper.model_preprocess,
        ])
        # unsqueeze(0) adds the batch dimension: [C, H, W] -> [1, C, H, W]
        self.test_image_tensor = preprocess(image).unsqueeze(0)

        # Unnormalized image for overlaying concept map heatmaps in plot()
        display_transform = transforms.Compose([
            transforms.Resize(
                (H, W),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop((H, W)),
        ])
        self.test_image_display = display_transform(image)

        print(f"Test image loaded: {os.path.basename(image_path)}")

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(self):
        """
        Run the model on the test image and return the top predictions.

        Returns
        -------
        Predictions
            Top-k predicted classes. Call .info() to print a formatted table.

        Raises
        ------
        RuntimeError
            If no test image has been set.
        """
        if self.test_image_tensor is None:
            raise RuntimeError(
                "test_image_path not set. Pass it to the constructor:\n"
                "  LocalVisualTCAV(model='resnet50', "
                "test_image_path='./image.jpg', ...)"
            )
        return super().predict(self.test_image_tensor, self.test_image_path)

    # -----------------------------------------------------------------------
    # Explanation
    # -----------------------------------------------------------------------

    def explain(
        self,
        cache_cav: bool = True,
        cache_random: bool = True,
    ) -> None:
        """
        Run the full Visual-TCAV explanation pipeline on the test image.

        Automatically calls predict() if it has not been called yet,
        so explicit predict() calls are never required before explain().

        For each (layer, concept) pair:
        1. Computes or loads random activations (reference distribution)
        2. Computes or loads the CAV
        3. Extracts feature maps from the test image
        4. Computes and normalizes the concept map
        5. Computes attribution scores for each target class

        Results are stored in self.computations and visualized with plot().

        Parameters
        ----------
        cache_cav : bool
            Save/load CAVs from disk to avoid recomputation. Default is True.
        cache_random : bool
            Save/load random activations from disk. Default is True.

        Raises
        ------
        RuntimeError
            If test image, concepts, or layers have not been configured.
        """
        # Validate configuration — errors here are informative and actionable
        if self.test_image_tensor is None:
            raise RuntimeError(
                "test_image_path not set. Pass it to the constructor:\n"
                "  LocalVisualTCAV(model='resnet50', "
                "test_image_path='./image.jpg', ...)"
            )
        if not self.concept_names:
            raise RuntimeError(
                "concept_names not set. Pass it to the constructor:\n"
                "  LocalVisualTCAV(..., concept_names=['striped', 'dotted'], "
                "concept_base_dir='./concept_images', ...)"
            )
        if not self.layer_names:
            raise RuntimeError(
                "layer_names not set. Pass it to the constructor:\n"
                "  LocalVisualTCAV(..., layer_names=['layer4'], ...)\n"
                "Call model_wrapper.info() to see available layer names."
            )

        # Auto-call predict() so the user is never blocked by a missing call
        if not self.target_classes:
            self.predict()

        print(f"\nRunning LocalVisualTCAV explanation...")
        print(f"  Image:    {os.path.basename(self.test_image_path)}")
        print(f"  Layers:   {self.layer_names}")
        print(f"  Concepts: {self.concept_names}")
        print(
            f"  Classes:  "
            f"{[self.model_wrapper.id_to_label(i) for i in self.target_classes]}\n"
        )

        for layer_name in self.layer_names:
            print(f"Layer: {layer_name}")

            random_activations = self._compute_random_activations(
                layer_name, use_cache=cache_random
            )

            feature_maps = self.model_wrapper.get_feature_maps(
                self.test_image_tensor, layer_name
            ).to(self.device)

            for concept_name in self.concept_names:
                print(f"  Concept: {concept_name}")

                cav = self._compute_cavs(
                    layer_name=layer_name,
                    concept_name=concept_name,
                    random_activations=random_activations,
                    use_cache=cache_cav,
                )
                self.computations[layer_name][concept_name].cav = cav

                raw_map = self._compute_concept_map(feature_maps, cav)
                concept_map = self._normalize_concept_map(raw_map, cav)
                self.computations[layer_name][concept_name].concept_map = (
                    concept_map.detach().cpu()
                )

                for class_index in self.target_classes:
                    attribution = self._compute_attribution(
                        feature_maps=feature_maps,
                        concept_map=concept_map,
                        cav=cav,
                        layer_name=layer_name,
                        class_index=class_index,
                    )
                    self.computations[layer_name][concept_name].attributions[
                        class_index
                    ] = attribution.detach().cpu()

                    class_name = self.model_wrapper.id_to_label(class_index)
                    print(f"    {class_name}: {attribution.item():.4f}")

        print("\nExplanation complete. Call plot() to visualize results.")

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot(
        self,
        colormap=None,
        figsize: tuple = None,
        save_path: str = None,
    ) -> None:
        """
        Visualize concept maps and attribution scores.

        Creates a grid with one row per concept:
        - Left columns: concept map heatmap overlaid on the original image.
          Red/yellow areas indicate high concept presence.
        - Last column: horizontal bar chart of attribution scores per class.

        Parameters
        ----------
        colormap : CustomColormap, optional
            Colormap for heatmaps. Defaults to DEFAULT_COLORMAP.
        figsize : tuple, optional
            Figure size as (width, height). Auto-computed if not provided.
        save_path : str, optional
            If provided, saves the figure to this path instead of displaying.

        Raises
        ------
        RuntimeError
            If explain() has not been called yet.
        """
        has_results = any(
            self.computations[layer][concept].concept_map is not None
            for layer in self.layer_names
            for concept in self.concept_names
        )
        if not has_results:
            raise RuntimeError(
                "No results found. Call explain() before plot()."
            )

        if colormap is None:
            colormap = DEFAULT_COLORMAP

        n_concepts = len(self.concept_names)
        n_layers = len(self.layer_names)
        n_cols = n_layers + 1

        if figsize is None:
            # Minimum readable size regardless of number of columns
            col_width = max(5, 16 // max(n_cols, 1))
            figsize = (n_cols * col_width, n_concepts * 5)

        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"Visual-TCAV — {os.path.basename(self.test_image_path)}",
            fontsize=14,
            fontweight="bold",
        )

        display_np = np.array(self.test_image_display)
        _, H, W = self.model_wrapper.input_size

        for row_idx, concept_name in enumerate(self.concept_names):
            for col_idx, layer_name in enumerate(self.layer_names):
                ax = fig.add_subplot(
                    n_concepts, n_cols, row_idx * n_cols + col_idx + 1
                )

                concept_map = (
                    self.computations[layer_name][concept_name].concept_map
                )

                # Bilinear upscaling gives a smooth overlay
                # (nearest-neighbor would produce a blocky 7x7 grid)
                concept_map_upscaled = F.interpolate(
                    concept_map.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()

                ax.imshow(display_np)
                colormap.imshow(concept_map_upscaled)
                ax.set_title(f"{concept_name}\n{layer_name}", fontsize=9)
                ax.axis("off")

            # Attribution bar chart (last column)
            ax_bar = fig.add_subplot(
                n_concepts, n_cols, row_idx * n_cols + n_layers + 1
            )

            class_labels = []
            attribution_values = []

            for class_index in self.target_classes:
                attribution = self.computations[
                    self.layer_names[-1]
                ][concept_name].attributions.get(class_index, 0.0)
                class_labels.append(
                    self.model_wrapper.id_to_label(class_index)
                )
                attribution_values.append(float(attribution))

            bars = ax_bar.barh(
                class_labels, attribution_values, color="steelblue"
            )
            ax_bar.set_xlabel("Attribution score")
            ax_bar.set_title(f"{concept_name}\nattributions", fontsize=9)
            max_val = max(attribution_values) if attribution_values else 1.0
            ax_bar.set_xlim(left=0, right=max(max_val * 1.2, 0.01))

            for bar, val in zip(bars, attribution_values):
                ax_bar.text(
                    bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center",
                    fontsize=8,
                )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()