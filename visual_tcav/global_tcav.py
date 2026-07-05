"""
global_tcav.py
--------------
GlobalVisualTCAV: explains a CLASS of images using Visual-TCAV.

Instead of explaining one single image (like LocalVisualTCAV),
GlobalVisualTCAV runs the explanation pipeline on a set of images
(e.g. 50 photos of zebras) and summarizes the results statistically.

For each (concept, layer, class) combination, it computes:
- The mean attribution score across all test images
- The standard deviation
- A 95% confidence interval

This tells you whether a concept consistently influences the model's
predictions for a given class, not just for one specific image.

Inherits all core logic from VisualTCAV base class.
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

# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class GlobalVisualTCAV(VisualTCAV):
    """
    Explains a class of images using Visual-TCAV.

    Runs the Visual-TCAV pipeline on a set of test images and
    summarizes attribution scores statistically, producing:
    - Mean attribution per (concept, layer, class)
    - Standard deviation
    - 95% confidence interval

    Parameters
    ----------
    model_wrapper : TorchModelWrapper
        The wrapped PyTorch model to explain.
    test_images_dir : str, optional
        Path to a folder containing test images (one class).
        Can also be set later with set_test_images_dir().
    concept_names : list of str, optional
        Names of the concepts to analyze.
    concept_base_dir : str, optional
        Path to folder containing one subfolder per concept.
    concept_dirs : dict, optional
        Dictionary mapping concept name to image folder path.
    random_dir : str, optional
        Path to folder containing random (negative) images.
    layer_names : list of str, optional
        CNN layers to analyze.
    n_classes : int, optional
        Number of top predicted classes to explain. Default is 3.
    m_steps : int, optional
        Interpolation steps for Integrated Gradients. Default is 50.
    max_examples : int, optional
        Maximum number of concept/random images to use. Default is 500.
    max_test_images : int, optional
        Maximum number of test images to process. Default is 50.
    cache_dir : str, optional
        Directory for caching CAVs and activations. Default is ".cache".

    Examples
    --------
    >>> import torchvision.models as models
    >>> from visual_tcav import GlobalVisualTCAV, TorchModelWrapper
    >>>
    >>> resnet = models.resnet50(weights='DEFAULT')
    >>> wrapper = TorchModelWrapper(model_name="resnet50", model=resnet)
    >>>
    >>> tcav = GlobalVisualTCAV(
    ...     model_wrapper=wrapper,
    ...     test_images_dir="./test_images/zebra",
    ...     concept_names=["striped", "dotted"],
    ...     concept_base_dir="./concept_images",
    ...     random_dir="./concept_images/random",
    ...     layer_names=["layer4"],
    ... )
    >>> tcav.explain()
    >>> tcav.statsInfo()
    >>> tcav.plot()
    """

    def __init__(
        self,
        model_wrapper,
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
    ):
        # Call parent constructor first
        super().__init__(
            model_wrapper=model_wrapper,
            n_classes=n_classes,
            m_steps=m_steps,
            max_examples=max_examples,
            cache_dir=cache_dir,
        )

        self.max_test_images = max_test_images

        # Test images — list of paths, one per image
        self.test_image_paths = []

        # Statistics storage:
        # stats[layer_name][concept_name][class_index] = Stat object
        self.stats = {}

        # Set test images dir if provided in constructor
        if test_images_dir is not None:
            self.set_test_images_dir(test_images_dir)

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

    def set_test_images_dir(self, images_dir: str) -> None:
        """
        Load all image paths from a folder.

        The folder should contain images of ONE class (e.g. 50 photos
        of zebras). Supported formats: JPG, PNG, BMP, TIFF, WEBP.

        Parameters
        ----------
        images_dir : str
            Path to folder containing test images.

        Raises
        ------
        FileNotFoundError
            If the folder does not exist.
        ValueError
            If the folder contains no valid images.

        Examples
        --------
        >>> tcav.set_test_images_dir("./test_images/zebra")
        """
        if not os.path.exists(images_dir):
            raise FileNotFoundError(
                f"Test images folder not found at: {images_dir}\n"
                f"Make sure the path is correct."
            )

        # Collect all image file paths from the folder
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

        # Limit to max_test_images
        self.test_image_paths = all_files[: self.max_test_images]

        print(
            f"Test images loaded: {len(self.test_image_paths)} images "
            f"from {images_dir}"
        )

    # -----------------------------------------------------------------------
    # Explanation
    # -----------------------------------------------------------------------

    def explain(
        self,
        cache_cav: bool = True,
        cache_random: bool = True,
    ) -> None:
        """
        Run the Visual-TCAV pipeline on all test images and compute
        attribution statistics.

        For each test image, runs the same pipeline as LocalVisualTCAV.
        Then aggregates all attribution scores into statistics:
        mean, standard deviation, and confidence interval.

        Results are stored in self.stats and can be visualized with
        statsInfo() and plot().

        Parameters
        ----------
        cache_cav : bool
            If True, saves/loads CAVs from disk. Default is True.
        cache_random : bool
            If True, saves/loads random activations. Default is True.

        Raises
        ------
        RuntimeError
            If test images, concepts, or layers have not been set.

        Examples
        --------
        >>> tcav.explain()
        >>> tcav.explain(cache_cav=False)  # recompute everything fresh
        """
        self._check_ready()

        print(f"\nRunning GlobalVisualTCAV explanation...")
        print(f"  Images:   {len(self.test_image_paths)} test images")
        print(f"  Layers:   {self.layer_names}")
        print(f"  Concepts: {self.concept_names}")
        print()

        # Storage for attribution scores per (layer, concept, class)
        # raw_attributions[layer][concept][class_index] = list of scores
        # one score per test image
        raw_attributions = {
            layer: {
                concept: {
                    class_idx: []
                    for class_idx in range(self.n_classes)
                }
                for concept in self.concept_names
            }
            for layer in self.layer_names
        }

        # ---------------------------------------------------------------
        # Phase 1: compute CAVs and random activations ONCE
        # (shared across all test images — no need to recompute per image)
        # ---------------------------------------------------------------
        print("Phase 1: Computing CAVs...")

        cavs = {}       # cavs[layer][concept] = Cav object
        fmaps_cache = {}  # will be filled per image in Phase 2

        for layer_name in self.layer_names:
            cavs[layer_name] = {}

            # Random activations: computed once per layer
            random_activations = self._compute_random_activations(
                layer_name, use_cache=cache_random
            )

            for concept_name in self.concept_names:
                cav = self._compute_cavs(
                    layer_name=layer_name,
                    concept_name=concept_name,
                    random_activations=random_activations,
                    use_cache=cache_cav,
                )
                cavs[layer_name][concept_name] = cav
                self.computations[layer_name][concept_name].cav = cav

        # ---------------------------------------------------------------
        # Phase 2: process each test image
        # ---------------------------------------------------------------
        print(f"\nPhase 2: Processing {len(self.test_image_paths)} images...")

        for img_path in tqdm(self.test_image_paths, desc="Images"):

            # Load and preprocess the image
            image_tensor = self._load_image(img_path)

            # Get predictions for this image
            predictions = super().predict(image_tensor, img_path)
            target_classes = [p.class_index for p in predictions.predictions[0]]

            # Store target classes from first image
            # (used later for display in statsInfo and plot)
            if not self.target_classes:
                self.target_classes = target_classes

            for layer_name in self.layer_names:

                # Extract feature maps for this image at this layer
                feature_maps = self.model_wrapper.get_feature_maps(
                    image_tensor, layer_name
                ).to(self.device)

                for concept_name in self.concept_names:
                    cav = cavs[layer_name][concept_name]

                    # Compute concept map for this image
                    raw_map = self._compute_concept_map(feature_maps, cav)
                    concept_map = self._normalize_concept_map(raw_map, cav)

                    # Compute attribution for each target class
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

        # ---------------------------------------------------------------
        # Phase 3: compute statistics from all attribution scores
        # ---------------------------------------------------------------
        print("\nPhase 3: Computing statistics...")

        self.stats = {}

        for layer_name in self.layer_names:
            self.stats[layer_name] = {}

            for concept_name in self.concept_names:
                self.stats[layer_name][concept_name] = {}

                for rank in range(self.n_classes):
                    scores = raw_attributions[layer_name][concept_name][rank]

                    if len(scores) > 0:
                        self.stats[layer_name][concept_name][rank] = Stat(
                            attributions=scores
                        )

        print("\nExplanation complete.")
        print("Call statsInfo() to see the summary table.")
        print("Call plot() to visualize the results.")

    # -----------------------------------------------------------------------
    # Results display
    # -----------------------------------------------------------------------

    def statsInfo(self) -> None:
        """
        Print a formatted table summarizing attribution statistics.

        For each (concept, layer, class) combination shows:
        - Mean attribution score
        - Standard deviation
        - 95% confidence interval [lower, upper]

        Must be called after explain().

        Raises
        ------
        RuntimeError
            If explain() has not been called yet.

        Examples
        --------
        >>> tcav.statsInfo()
        +----------+---------+-------+--------+------------------+
        | Concept  |  Layer  | Class |  Mean  |   95% CI         |
        +----------+---------+-------+--------+------------------+
        | striped  | layer4  | zebra | 0.2290 | [0.210, 0.248]   |
        | dotted   | layer4  | zebra | 0.0210 | [0.018, 0.024]   |
        +----------+---------+-------+--------+------------------+
        """
        self._check_explained()

        from prettytable import PrettyTable

        table = PrettyTable(
            title=f"GlobalVisualTCAV — {self.model_wrapper.model_name}",
            field_names=[
                "Concept",
                "Layer",
                "Class",
                "Mean",
                "Std",
                "95% CI",
            ],
        )
        table.float_format = ".4"
        table.align = "l"

        for layer_name in self.layer_names:
            for concept_name in self.concept_names:
                for rank in range(self.n_classes):

                    if rank not in self.stats[layer_name][concept_name]:
                        continue

                    stat = self.stats[layer_name][concept_name][rank]

                    # Get class name for this rank
                    if rank < len(self.target_classes):
                        class_name = self.model_wrapper.id_to_label(
                            self.target_classes[rank]
                        )
                    else:
                        class_name = f"class_{rank}"

                    ci_lower = f"{stat.begin.item():.4f}"
                    ci_upper = f"{stat.end.item():.4f}"

                    table.add_row([
                        concept_name,
                        layer_name,
                        class_name,
                        f"{stat.mean.item():.4f}",
                        f"{stat.std.item():.4f}",
                        f"[{ci_lower}, {ci_upper}]",
                    ])

        print(table)

    def plot(
        self,
        figsize: tuple = None,
        save_path: str = None,
    ) -> None:
        """
        Visualize global attribution statistics as bar charts.

        Creates one subplot per layer, showing mean attribution scores
        for each concept across the top predicted classes.
        Error bars represent the 95% confidence interval.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size as (width, height). Auto-computed if not provided.
        save_path : str, optional
            If provided, saves the figure to this path.

        Raises
        ------
        RuntimeError
            If explain() has not been called yet.

        Examples
        --------
        >>> tcav.plot()
        >>> tcav.plot(save_path="./results/global_zebra.png")
        """
        self._check_explained()

        n_layers = len(self.layer_names)
        n_concepts = len(self.concept_names)

        if figsize is None:
            figsize = (n_layers * 6, 5)

        fig, axes = plt.subplots(1, n_layers, figsize=figsize)

        # Handle case of single layer (axes is not a list then)
        if n_layers == 1:
            axes = [axes]

        fig.suptitle(
            f"GlobalVisualTCAV — {self.model_wrapper.model_name}",
            fontsize=14,
            fontweight="bold",
        )

        # Color palette for concepts
        colors = plt.cm.tab10(np.linspace(0, 1, n_concepts))

        for ax, layer_name in zip(axes, self.layer_names):

            # For each class (rank 0 = top predicted class)
            # show grouped bars: one group per class, one bar per concept
            n_classes_to_show = self.n_classes
            x = np.arange(n_classes_to_show)
            bar_width = 0.8 / n_concepts

            for concept_idx, concept_name in enumerate(self.concept_names):

                means = []
                errors = []

                for rank in range(n_classes_to_show):
                    if rank in self.stats[layer_name][concept_name]:
                        stat = self.stats[layer_name][concept_name][rank]
                        means.append(stat.mean.item())
                        # Error bar = half the confidence interval width
                        errors.append(
                            (stat.end.item() - stat.begin.item()) / 2
                        )
                    else:
                        means.append(0.0)
                        errors.append(0.0)

                # Position of bars for this concept
                offset = (concept_idx - n_concepts / 2) * bar_width + bar_width / 2

                ax.bar(
                    x + offset,
                    means,
                    bar_width,
                    label=concept_name,
                    color=colors[concept_idx],
                    yerr=errors,
                    capsize=4,
                    error_kw={"elinewidth": 1.5},
                )

            # X axis labels: class names
            class_labels = []
            for rank in range(n_classes_to_show):
                if rank < len(self.target_classes):
                    class_labels.append(
                        self.model_wrapper.id_to_label(self.target_classes[rank])
                    )
                else:
                    class_labels.append(f"class_{rank}")

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
        """
        Load and preprocess a single image from disk.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        torch.Tensor
            Preprocessed image tensor. Shape: [1, C, H, W].
        """
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

    def _check_ready(self) -> None:
        """Raise RuntimeError if not ready to run explain()."""
        if not self.test_image_paths:
            raise RuntimeError(
                "No test images set. Call set_test_images_dir() first, "
                "or pass test_images_dir to the constructor."
            )
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

    def _check_explained(self) -> None:
        """Raise RuntimeError if explain() has not been called yet."""
        if not self.stats:
            raise RuntimeError(
                "No results found. Call explain() before "
                "statsInfo() or plot()."
            )