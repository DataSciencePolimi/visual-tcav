"""
global_tcav.py
--------------
GlobalVisualTCAV: explains a class of images using Visual-TCAV.

Runs the explanation pipeline on a set of test images and summarizes
attribution scores statistically (mean, std, 95% confidence interval),
answering whether a concept consistently influences predictions for a class.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from visual_tcav.visual_tcav import VisualTCAV
from visual_tcav.utils import Stat

sys.dont_write_bytecode = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class GlobalVisualTCAV(VisualTCAV):
    """
    Explains a class of images using Visual-TCAV.

    Runs the pipeline on a folder of test images and summarizes attribution
    scores statistically, answering:

    **Does this concept consistently influence predictions for this class?**

    Before instantiating, use :func:`~visual_tcav.available_layers` to
    inspect available layer names:

    .. code-block:: python

        from visual_tcav import available_layers, GlobalVisualTCAV

        available_layers("resnet50")

        tcav = GlobalVisualTCAV(
            model="resnet50",
            test_images_dir="./images/zebra",
            concept_names=["striped", "dotted"],
            concept_base_dir="./concept_images",
            random_dir="./concept_images/random",
            layer_names=["layer4"],
        )
        tcav.explain()
        tcav.statsInfo()
        tcav.plot()

    All parameters are optional at construction time. Validation happens
    at explain() time with clear, actionable error messages.

    Parameters
    ----------
    model : str or nn.Module, optional
        Model name string or PyTorch model object.
    model_name : str, optional
        Display name for the model.
    model_wrapper : TorchModelWrapper, optional
        Pre-built wrapper for advanced use cases.
    test_images_dir : str, optional
        Folder containing test images of ONE class.
    concept_names : list of str, optional
        Names of the concepts to analyze.
    concept_base_dir : str, optional
        Folder containing one subfolder per concept.
    concept_dirs : dict, optional
        Explicit mapping of concept name to image folder.
    random_dir : str, optional
        Folder containing random images (reference distribution).
    layer_names : list of str, optional
        CNN layers to analyze.
    n_classes : int, optional
        Number of top predicted classes. Default is 3.
    m_steps : int, optional
        Interpolation steps for IG. Default is 50.
    max_examples : int, optional
        Maximum concept/random images. Default is 500.
    max_test_images : int, optional
        Maximum test images to process. Default is 50.
    cache_dir : str, optional
        Directory for caching. Default is ".cache".
        Set to None to disable caching.
    cav_fn : callable, optional
        Custom CAV computation function.
    """

    def __init__(
        self,
        model=None,
        model_name: str = None,
        model_wrapper=None,
        test_images_dir: str = None,
        concept_names: list = None,
        concept_base_dir: str = None,
        concept_dirs: dict = None,
        random_dir: str = None,
        layer_names: list = None,
        n_classes: int = 3,
        m_steps: int = 50,
        max_examples: int = 500,
        max_test_images: int = 50,
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

        self.max_test_images = max_test_images
        self.test_image_paths = []
        # stats[layer][concept][rank] = Stat object
        self.stats = {}

        if test_images_dir is not None:
            self._load_test_images(test_images_dir)

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

    def _load_test_images(self, images_dir: str) -> None:
        """
        Load all image paths from a folder.

        Parameters
        ----------
        images_dir : str
            Folder containing test images of ONE class.

        Raises
        ------
        FileNotFoundError
            If the folder does not exist.
        ValueError
            If no valid images are found.
        """
        if not os.path.exists(images_dir):
            raise FileNotFoundError(
                f"Test images folder not found at: {images_dir}"
            )

        all_files = [
            os.path.join(images_dir, f)
            for f in sorted(os.listdir(images_dir))
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]

        if not all_files:
            raise ValueError(
                f"No valid images found in: {images_dir}\n"
                f"Supported formats: {IMAGE_EXTENSIONS}"
            )

        self.test_image_paths = all_files[: self.max_test_images]
        print(
            f"Test images loaded: {len(self.test_image_paths)} "
            f"from {images_dir}"
        )

    # -----------------------------------------------------------------------
    # Explanation
    # -----------------------------------------------------------------------

    def explain(self, force_recompute: bool = False) -> None:
        """
        Run the Visual-TCAV pipeline on all test images and compute statistics.

        Three-phase pipeline:

        **Phase 1** — Compute CAVs once per (layer, concept) pair. CAVs only
        depend on concept images, not test images, so they are reused
        across all test images.

        **Phase 2** — Process each test image: extract feature maps, compute
        concept map and attribution scores, append to raw score lists.

        **Phase 3** — Wrap each list of scores in a Stat object
        (mean, std, 95% confidence interval).

        Parameters
        ----------
        force_recompute : bool
            If True, ignores all cached results and recomputes from scratch.
            If False (default), loads cached CAVs and activations when
            available to avoid redundant computation.

        Raises
        ------
        RuntimeError
            If test images, concepts, or layers have not been configured.

        Examples
        --------
        >>> tcav.explain()                     # use cache (default, fast)
        >>> tcav.explain(force_recompute=True)  # recompute everything fresh
        """
        if not self.test_image_paths:
            raise RuntimeError(
                "test_images_dir not set. Pass it to the constructor:\n"
                "  GlobalVisualTCAV(model='resnet50', "
                "test_images_dir='./images/zebra', ...)"
            )
        if not self.concept_names:
            raise RuntimeError(
                "concept_names not set. Pass it to the constructor:\n"
                "  GlobalVisualTCAV(..., concept_names=['striped'], "
                "concept_base_dir='./concept_images', ...)"
            )
        if not self.layer_names:
            raise RuntimeError(
                "layer_names not set. Pass it to the constructor:\n"
                "  GlobalVisualTCAV(..., layer_names=['layer4'], ...)\n"
                "Use available_layers(model) to see valid layer names."
            )

        print(f"\nRunning GlobalVisualTCAV explanation...")
        print(f"  Images:   {len(self.test_image_paths)} test images")
        print(f"  Layers:   {self.layer_names}")
        print(f"  Concepts: {self.concept_names}\n")

        # raw_attributions[layer][concept][rank] = list of scores
        # Rank is used instead of class index because different images may have
        # slightly different top predictions; rank ensures consistent comparison
        raw_attributions = {
            layer: {
                concept: {rank: [] for rank in range(self.n_classes)}
                for concept in self.concept_names
            }
            for layer in self.layer_names
        }

        # Phase 1: compute CAVs once — reused across all test images
        print("Phase 1: Computing CAVs...")
        cavs = {}
        for layer_name in self.layer_names:
            cavs[layer_name] = {}
            random_activations = self._compute_random_activations(
                layer_name, force_recompute=force_recompute
            )
            for concept_name in self.concept_names:
                cav = self._compute_cavs(
                    layer_name=layer_name,
                    concept_name=concept_name,
                    random_activations=random_activations,
                    force_recompute=force_recompute,
                )
                cavs[layer_name][concept_name] = cav
                self.computations[layer_name][concept_name].cav = cav

        # Phase 2: process each test image
        print(f"\nPhase 2: Processing {len(self.test_image_paths)} images...")
        for img_path in tqdm(self.test_image_paths, desc="Images"):
            image_tensor = self._load_image(img_path)
            predictions = super().predict(image_tensor, img_path)
            target_classes = [p.class_index for p in predictions.predictions[0]]

            if not self.target_classes:
                self.target_classes = target_classes

            for layer_name in self.layer_names:
                feature_maps = self.model_wrapper.get_feature_maps(
                    image_tensor, layer_name
                ).to(self.device)

                for concept_name in self.concept_names:
                    cav = cavs[layer_name][concept_name]
                    raw_map = self._compute_concept_map(feature_maps, cav)
                    concept_map = self._normalize_concept_map(raw_map, cav)

                    for rank, class_index in enumerate(target_classes):
                        attribution = self._compute_attribution(
                            feature_maps=feature_maps,
                            concept_map=concept_map,
                            cav=cav,
                            layer_name=layer_name,
                            class_index=class_index,
                        )
                        raw_attributions[layer_name][concept_name][rank].append(
                            attribution.item()
                        )

        # Phase 3: compute statistics
        print("\nPhase 3: Computing statistics...")
        self.stats = {}
        for layer_name in self.layer_names:
            self.stats[layer_name] = {}
            for concept_name in self.concept_names:
                self.stats[layer_name][concept_name] = {}
                for rank in range(self.n_classes):
                    scores = raw_attributions[layer_name][concept_name][rank]
                    if scores:
                        self.stats[layer_name][concept_name][rank] = Stat(scores)

        print("\nExplanation complete. Call statsInfo() or plot() for results.")

    # -----------------------------------------------------------------------
    # Results display
    # -----------------------------------------------------------------------

    def statsInfo(self) -> None:
        """
        Print attribution statistics across all test images.

        Raises
        ------
        RuntimeError
            If explain() has not been called yet.
        """
        if not self.stats:
            raise RuntimeError(
                "No results found. Call explain() before statsInfo()."
            )

        from prettytable import PrettyTable

        table = PrettyTable(
            title=f"GlobalVisualTCAV — {self.model_wrapper.model_name}",
            field_names=["Concept", "Layer", "Class", "Mean", "Std", "95% CI"],
        )
        table.float_format = ".4"
        table.align = "l"

        for layer_name in self.layer_names:
            for concept_name in self.concept_names:
                for rank in range(self.n_classes):
                    if rank not in self.stats[layer_name][concept_name]:
                        continue

                    stat = self.stats[layer_name][concept_name][rank]
                    class_name = (
                        self.model_wrapper.id_to_label(self.target_classes[rank])
                        if rank < len(self.target_classes)
                        else f"class_{rank}"
                    )

                    table.add_row([
                        concept_name, layer_name, class_name,
                        f"{stat.mean.item():.4f}",
                        f"{stat.std.item():.4f}",
                        f"[{stat.begin.item():.4f}, {stat.end.item():.4f}]",
                    ])

        print(table)

    def plot(self, figsize: tuple = None, save_path: str = None) -> None:
        """
        Visualize global attribution statistics as grouped bar charts.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size as (width, height). Auto-computed if not provided.
        save_path : str, optional
            If provided, saves the figure to disk.

        Raises
        ------
        RuntimeError
            If explain() has not been called yet.
        """
        if not self.stats:
            raise RuntimeError(
                "No results found. Call explain() before plot()."
            )

        n_layers = len(self.layer_names)
        n_concepts = len(self.concept_names)

        if figsize is None:
            figsize = (max(n_layers * 6, 8), 5)

        fig, axes = plt.subplots(1, n_layers, figsize=figsize)
        if n_layers == 1:
            axes = [axes]

        fig.suptitle(
            f"GlobalVisualTCAV — {self.model_wrapper.model_name}",
            fontsize=14, fontweight="bold",
        )

        colors = plt.cm.tab10(np.linspace(0, 1, n_concepts))

        for ax, layer_name in zip(axes, self.layer_names):
            x = np.arange(self.n_classes)
            bar_width = 0.8 / n_concepts

            for concept_idx, concept_name in enumerate(self.concept_names):
                means, errors = [], []
                for rank in range(self.n_classes):
                    if rank in self.stats[layer_name][concept_name]:
                        stat = self.stats[layer_name][concept_name][rank]
                        means.append(stat.mean.item())
                        errors.append(
                            (stat.end.item() - stat.begin.item()) / 2
                        )
                    else:
                        means.append(0.0)
                        errors.append(0.0)

                offset = (concept_idx - n_concepts / 2) * bar_width + bar_width / 2
                ax.bar(
                    x + offset, means, bar_width,
                    label=concept_name, color=colors[concept_idx],
                    yerr=errors, capsize=4, error_kw={"elinewidth": 1.5},
                )

            class_labels = [
                self.model_wrapper.id_to_label(self.target_classes[r])
                if r < len(self.target_classes) else f"class_{r}"
                for r in range(self.n_classes)
            ]

            ax.set_xticks(x)
            ax.set_xticklabels(class_labels, rotation=15, ha="right")
            ax.set_ylabel("Mean attribution score")
            ax.set_title(f"Layer: {layer_name}")
            ax.set_ylim(bottom=0)
            ax.legend(title="Concepts", loc="upper right")
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image. Returns [1, C, H, W] tensor."""
        from PIL import Image
        from torchvision import transforms

        image = Image.open(image_path).convert("RGB")
        C, H, W = self.model_wrapper.input_size

        preprocess = transforms.Compose([
            transforms.Resize(
                (H, W),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop((H, W)),
            transforms.ToTensor(),
            self.model_wrapper.model_preprocess,
        ])

        return preprocess(image).unsqueeze(0)