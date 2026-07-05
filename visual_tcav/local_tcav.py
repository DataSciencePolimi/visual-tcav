"""
local_tcav.py
-------------
LocalVisualTCAV: explains a single test image using Visual-TCAV.

Given one image and a set of concepts, computes:
- A concept map (heatmap) for each (concept, layer) pair
- An attribution score for each (concept, layer, class) combination

Inherits all core logic from VisualTCAV base class.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visual_tcav.visual_tcav import VisualTCAV
from visual_tcav.utils import DEFAULT_COLORMAP

sys.dont_write_bytecode = True


class LocalVisualTCAV(VisualTCAV):
    """
    Explains a single test image using Visual-TCAV.

    For each combination of (concept, layer), computes:
    - A concept map: a heatmap showing WHERE in the image the CNN
      detected the concept
    - An attribution score: a number showing HOW MUCH the concept
      contributed to each predicted class

    Parameters
    ----------
    model_wrapper : TorchModelWrapper
        The wrapped PyTorch model to explain.
    test_image_path : str, optional
        Path to the image to explain. Can also be set later with
        set_test_image().
    concept_names : list of str, optional
        Names of the concepts to analyze. Can also be set later with
        set_concepts().
    concept_base_dir : str, optional
        Path to folder containing one subfolder per concept.
    random_dir : str, optional
        Path to folder containing random (negative) images.
    layer_names : list of str, optional
        CNN layers to analyze. Can also be set later with set_layers().
    n_classes : int, optional
        Number of top predicted classes to explain. Default is 3.
    m_steps : int, optional
        Interpolation steps for Integrated Gradients. Default is 50.
    max_examples : int, optional
        Maximum number of concept/random images to use. Default is 500.
    cache_dir : str, optional
        Directory for caching CAVs and activations. Default is ".cache".

    Examples
    --------
    >>> import torchvision.models as models
    >>> from visual_tcav import LocalVisualTCAV, TorchModelWrapper
    >>>
    >>> resnet = models.resnet50(weights='DEFAULT')
    >>> wrapper = TorchModelWrapper(model_name="resnet50", model=resnet)
    >>>
    >>> tcav = LocalVisualTCAV(
    ...     model_wrapper=wrapper,
    ...     test_image_path="./zebra.jpg",
    ...     concept_names=["striped", "dotted"],
    ...     concept_base_dir="./concept_images",
    ...     random_dir="./concept_images/random",
    ...     layer_names=["layer4"],
    ... )
    >>> tcav.predict().info()
    >>> tcav.explain()
    >>> tcav.plot()
    """

    def __init__(
        self,
        model_wrapper,
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
    ):
        # Call the parent class constructor first
        # This sets up model_wrapper, n_classes, m_steps, etc.
        super().__init__(
            model_wrapper=model_wrapper,
            n_classes=n_classes,
            m_steps=m_steps,
            max_examples=max_examples,
            cache_dir=cache_dir,
        )

        # Test image — will be set by set_test_image()
        self.test_image_path = None
        self.test_image_tensor = None   # preprocessed tensor for the model
        self.test_image_display = None  # original image for display in plots

        # Set test image if provided in constructor
        if test_image_path is not None:
            self.set_test_image(test_image_path)

        # Set concepts if provided in constructor
        if concept_names is not None:
            self.set_concepts(
                concept_names=concept_names,
                concept_dirs=concept_dirs,
                concept_base_dir=concept_base_dir,
                random_dir=random_dir,
            )

        # Set layers if provided in constructor
        if layer_names is not None:
            self.set_layers(layer_names)

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def set_test_image(self, image_path: str) -> None:
        """
        Load and preprocess a test image.

        Loads the image from disk and creates two versions:
        - A preprocessed tensor for the model (normalized, resized)
        - The original image for display in plots

        Parameters
        ----------
        image_path : str
            Path to the image file. Supports JPEG, PNG, and other
            formats supported by PIL.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist at the given path.

        Examples
        --------
        >>> tcav.set_test_image("./images/zebra.jpg")
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Test image not found at: {image_path}\n"
                f"Make sure the path is correct."
            )

        self.test_image_path = image_path

        # Load image with PIL
        image = Image.open(image_path).convert("RGB")

        # Version 1: preprocessed tensor for the model
        # Resize, crop, convert to tensor, normalize
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
        self.test_image_tensor = preprocess(image).unsqueeze(0)
        # Shape: [1, C, H, W] — batch dimension added with unsqueeze(0)

        # Version 2: resized image for display (no normalization)
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

        Must be called after set_test_image().

        Returns
        -------
        Predictions
            Object containing top-k predicted classes with names and
            confidence scores. Call .info() on the result to print
            a formatted table.

        Raises
        ------
        RuntimeError
            If no test image has been set.

        Examples
        --------
        >>> tcav.predict().info()
        +---------------------------+
        | Model: resnet50           |
        +-----------+---------------+------------+
        | Image     | Class name    | Confidence |
        +-----------+---------------+------------+
        | zebra.jpg | zebra         | 0.9900     |
        |           | tiger         | 0.0010     |
        |           | cat           | 0.0005     |
        +-----------+---------------+------------+
        """
        self._check_test_image()
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

        For each combination of (layer, concept):
        1. Computes or loads random activations (negative examples)
        2. Computes or loads the CAV
        3. Extracts feature maps from the test image
        4. Computes the concept map (WHERE is the concept?)
        5. Computes attribution scores (HOW MUCH does the concept matter?)

        Results are stored in self.computations and can be visualized
        with plot().

        Parameters
        ----------
        cache_cav : bool
            If True, saves/loads CAVs from disk to avoid recomputing.
            Default is True.
        cache_random : bool
            If True, saves/loads random activations from disk.
            Default is True.

        Raises
        ------
        RuntimeError
            If test image, concepts, or layers have not been set.

        Examples
        --------
        >>> tcav.explain()
        >>> tcav.explain(cache_cav=False)  # recompute everything fresh
        """
        self._check_ready()

        print(f"\nRunning LocalVisualTCAV explanation...")
        print(f"  Image:    {os.path.basename(self.test_image_path)}")
        print(f"  Layers:   {self.layer_names}")
        print(f"  Concepts: {self.concept_names}")
        print(f"  Classes:  {[self.model_wrapper.id_to_label(i) for i in self.target_classes]}")
        print()

        for layer_name in self.layer_names:
            print(f"Layer: {layer_name}")

            # Step 1: Compute random activations (shared across all concepts)
            random_activations = self._compute_random_activations(
                layer_name, use_cache=cache_random
            )

            # Step 2: Extract feature maps for the test image at this layer
            # Shape: [1, C, H, W]
            feature_maps = self.model_wrapper.get_feature_maps(
                self.test_image_tensor, layer_name
            ).to(self.device)

            for concept_name in self.concept_names:
                print(f"  Concept: {concept_name}")

                # Step 3: Compute CAV for this (concept, layer) pair
                cav = self._compute_cavs(
                    layer_name=layer_name,
                    concept_name=concept_name,
                    random_activations=random_activations,
                    use_cache=cache_cav,
                )
                self.computations[layer_name][concept_name].cav = cav

                # Step 4: Compute raw concept map
                raw_map = self._compute_concept_map(feature_maps, cav)

                # Step 5: Normalize concept map to [0, 1]
                concept_map = self._normalize_concept_map(raw_map, cav)
                self.computations[layer_name][concept_name].concept_map = (
                    concept_map.detach().cpu()
                )

                # Step 6: Compute attribution score for each target class
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
                    print(f"    Attribution for '{class_name}': "
                          f"{attribution.item():.4f}")

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
        Visualize the explanation results.

        Creates a figure with:
        - One row per concept
        - One column per layer, showing the concept map overlaid
          on the original image
        - A bar chart showing attribution scores for each class

        Parameters
        ----------
        colormap : CustomColormap, optional
            Colormap for the heatmaps. Defaults to DEFAULT_COLORMAP.
        figsize : tuple, optional
            Figure size as (width, height). Auto-computed if not provided.
        save_path : str, optional
            If provided, saves the figure to this path instead of
            displaying it.

        Raises
        ------
        RuntimeError
            If explain() has not been called yet.

        Examples
        --------
        >>> tcav.plot()
        >>> tcav.plot(save_path="./results/zebra_explanation.png")
        """
        self._check_explained()

        if colormap is None:
            colormap = DEFAULT_COLORMAP

        n_concepts = len(self.concept_names)
        n_layers = len(self.layer_names)
        n_classes = len(self.target_classes)

        # Each concept gets: n_layers concept map columns + 1 attribution column
        n_cols = n_layers + 1
        n_rows = n_concepts

        if figsize is None:
            figsize = (n_cols * 4, n_rows * 4)

        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"Visual-TCAV — {os.path.basename(self.test_image_path)}",
            fontsize=14,
            fontweight="bold",
        )

        # Convert display image to numpy for matplotlib
        display_np = np.array(self.test_image_display)
        _, H, W = self.model_wrapper.input_size

        for row_idx, concept_name in enumerate(self.concept_names):
            for col_idx, layer_name in enumerate(self.layer_names):

                ax = fig.add_subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)

                # Get concept map for this (concept, layer) pair
                concept_map = self.computations[layer_name][concept_name].concept_map

                # Upscale concept map from [H_small, W_small] to [H, W]
                # so it can be overlaid on the original image
                concept_map_upscaled = F.interpolate(
                    concept_map.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()

                # Show original image
                ax.imshow(display_np)

                # Overlay concept map heatmap
                colormap.imshow(concept_map_upscaled)

                ax.set_title(f"{concept_name}\n{layer_name}", fontsize=9)
                ax.axis("off")

            # Attribution bar chart — last column for this concept row
            ax_bar = fig.add_subplot(
                n_rows, n_cols, row_idx * n_cols + n_layers + 1
            )

            # Collect attribution scores for each class
            class_labels = []
            attribution_values = []

            for class_index in self.target_classes:
                attribution = self.computations[
                    self.layer_names[-1]
                ][concept_name].attributions.get(class_index, 0.0)

                class_labels.append(
                    self.model_wrapper.id_to_label(class_index)
                )
                attribution_values.append(
                    float(attribution) if torch.is_tensor(attribution)
                    else float(attribution)
                )

            # Draw horizontal bar chart
            bars = ax_bar.barh(
                class_labels,
                attribution_values,
                color="steelblue",
            )
            ax_bar.set_xlabel("Attribution score")
            ax_bar.set_title(f"{concept_name}\nattributions", fontsize=9)
            ax_bar.set_xlim(left=0)

            # Add value labels on bars
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

    # -----------------------------------------------------------------------
    # Private validation helpers
    # -----------------------------------------------------------------------

    def _check_test_image(self) -> None:
        """Raise RuntimeError if no test image has been set."""
        if self.test_image_tensor is None:
            raise RuntimeError(
                "No test image set. Call set_test_image() first, "
                "or pass test_image_path to the constructor."
            )

    def _check_ready(self) -> None:
        """Raise RuntimeError if not ready to run explain()."""
        self._check_test_image()

        if not self.concept_names:
            raise RuntimeError(
                "No concepts set. Call set_concepts() first, "
                "or pass concept_names to the constructor."
            )
        if not self.layer_names:
            raise RuntimeError(
                "No layers set. Call set_layers() first, "
                "or pass layer_names to the constructor."
            )
        if not self.target_classes:
            raise RuntimeError(
                "No target classes found. Call predict() before explain()."
            )

    def _check_explained(self) -> None:
        """Raise RuntimeError if explain() has not been called yet."""
        has_results = any(
            self.computations[layer][concept].concept_map is not None
            for layer in self.layer_names
            for concept in self.concept_names
        )
        if not has_results:
            raise RuntimeError(
                "No results found. Call explain() before plot()."
            )