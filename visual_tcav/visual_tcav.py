"""
visual_tcav.py
--------------
Base class for Visual-TCAV. Contains all shared logic for computing
CAVs, integrated gradients, and concept maps.

Both LocalVisualTCAV and GlobalVisualTCAV inherit from this class.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from joblib import dump, load

from visual_tcav.utils import (
    Cav,
    ConceptLayer,
    Prediction,
    Predictions,
    contraharmonic_mean,
    nth_highest_index,
)

sys.dont_write_bytecode = True


class VisualTCAV:
    """
    Base class for Visual-TCAV.

    Implements the core pipeline shared by both the Local and Global
    explainers:
    - Managing concepts and layers
    - Computing Concept Activation Vectors (CAVs)
    - Computing random activations (negative examples)
    - Computing Integrated Gradients
    - Computing concept maps

    This class is not meant to be used directly. Use LocalVisualTCAV
    or GlobalVisualTCAV instead.

    Parameters
    ----------
    model_wrapper : TorchModelWrapper
        The wrapped PyTorch model to explain.
    n_classes : int, optional
        Number of top predicted classes to explain. Default is 3.
    m_steps : int, optional
        Number of interpolation steps for Integrated Gradients.
        Higher = more accurate but slower. Default is 50.
    max_examples : int, optional
        Maximum number of concept/random images to use when computing
        CAVs. Default is 500.
    cache_dir : str, optional
        Directory where CAVs and activations are cached to disk.
        Default is ".cache".

    Examples
    --------
    Do not use this class directly. Use LocalVisualTCAV instead:

    >>> from visual_tcav import LocalVisualTCAV, TorchModelWrapper
    >>> import torchvision.models as models
    >>> resnet = models.resnet50(weights='DEFAULT')
    >>> wrapper = TorchModelWrapper(model_name="resnet50", model=resnet)
    >>> tcav = LocalVisualTCAV(model_wrapper=wrapper)
    """

    def __init__(
        self,
        model_wrapper,
        n_classes: int = 3,
        m_steps: int = 50,
        max_examples: int = 500,
        cache_dir: str = ".cache",
    ):
        self.model_wrapper = model_wrapper
        self.n_classes = n_classes
        self.m_steps = m_steps
        self.max_examples = max_examples
        self.cache_dir = cache_dir

        # Will be set by set_concepts() and set_layers()
        self.concept_names = []
        self.concept_dirs = {}
        self.layer_names = []
        self.random_dir = None

        # Target classes determined after predict()
        self.target_classes = []

        # Main storage: computations[layer_name][concept_name] = ConceptLayer
        self.computations = {}

        # Device: GPU if available, otherwise CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create cache directory if it does not exist
        os.makedirs(self.cache_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public setup methods
    # -----------------------------------------------------------------------

    def set_concepts(
        self,
        concept_names: list,
        concept_dirs: dict = None,
        concept_base_dir: str = None,
        random_dir: str = None,
    ) -> None:
        """
        Specify which concepts to analyze.

        You can either provide a dictionary mapping concept names to
        their image folders, or provide a base directory that contains
        one subfolder per concept.

        Parameters
        ----------
        concept_names : list of str
            Names of the concepts to analyze (e.g. ["striped", "dotted"]).
        concept_dirs : dict, optional
            Dictionary mapping concept name → path to image folder.
            Example: {"striped": "./images/striped", "dotted": "./images/dotted"}
        concept_base_dir : str, optional
            Path to a folder that contains one subfolder per concept.
            Example: if concept_base_dir="./images" and concept_names=["striped"],
            it will look for images in "./images/striped/".
        random_dir : str, optional
            Path to the folder containing random (negative) images.
            If not provided, looks for a "random" folder inside concept_base_dir.

        Raises
        ------
        ValueError
            If neither concept_dirs nor concept_base_dir is provided.
        ValueError
            If any concept folder does not exist.

        Examples
        --------
        >>> tcav.set_concepts(
        ...     concept_names=["striped", "dotted"],
        ...     concept_base_dir="./concept_images",
        ...     random_dir="./concept_images/random",
        ... )
        """
        if concept_dirs is not None:
            # User provided explicit paths
            self.concept_dirs = concept_dirs
        elif concept_base_dir is not None:
            # Build paths automatically from base directory
            self.concept_dirs = {
                name: os.path.join(concept_base_dir, name)
                for name in concept_names
            }
        else:
            raise ValueError(
                "You must provide either 'concept_dirs' (a dictionary mapping "
                "concept names to folders) or 'concept_base_dir' (a folder "
                "containing one subfolder per concept)."
            )

        # Validate that all concept folders exist
        for name, path in self.concept_dirs.items():
            if not os.path.exists(path):
                raise ValueError(
                    f"Concept folder for '{name}' not found at: {path}\n"
                    f"Make sure the folder exists and contains images."
                )

        self.concept_names = concept_names

        # Set random images directory
        if random_dir is not None:
            self.random_dir = random_dir
        elif concept_base_dir is not None:
            self.random_dir = os.path.join(concept_base_dir, "random")
        else:
            self.random_dir = None

        if self.random_dir and not os.path.exists(self.random_dir):
            raise ValueError(
                f"Random images folder not found at: {self.random_dir}\n"
                f"Random images are needed as negative examples for CAV computation."
            )

        print(f"Concepts set: {self.concept_names}")

    def set_layers(self, layer_names: list) -> None:
        """
        Specify which CNN layers to analyze.

        To see which layers are available, call model_wrapper.info() first.
        Deeper layers (closer to the output) tend to represent higher-level
        concepts. For ResNet50, 'layer4' is a good starting point.

        Parameters
        ----------
        layer_names : list of str
            Names of the layers to analyze.
            Example: ["layer3", "layer4"]

        Raises
        ------
        ValueError
            If no layer names are provided.

        Examples
        --------
        >>> tcav.set_layers(["layer4"])
        >>> tcav.set_layers(["layer3", "layer4"])
        """
        if not layer_names:
            raise ValueError(
                "You must provide at least one layer name. "
                "Call model_wrapper.info() to see available layers."
            )
        self.layer_names = layer_names

        # Initialize the computations dictionary
        # computations[layer][concept] = ConceptLayer()
        self.computations = {
            layer: {
                concept: ConceptLayer()
                for concept in self.concept_names
            }
            for layer in self.layer_names
        }

        print(f"Layers set: {self.layer_names}")

    # -----------------------------------------------------------------------
    # CAV computation
    # -----------------------------------------------------------------------

    def _compute_random_activations(
        self, layer_name: str, use_cache: bool = True
    ) -> torch.Tensor:
        """
        Compute (or load from cache) the pooled activations of random images
        at the specified layer.

        Random images serve as the NEGATIVE examples when computing the CAV.
        They represent "everything that is NOT the concept".

        The activations are pooled with Global Average Pooling (GAP):
        instead of keeping the full [C, H, W] feature map for each image,
        we average across H and W to get a single vector of size [C].
        This gives one vector per image, which is then used to compute
        the negative centroid.

        Parameters
        ----------
        layer_name : str
            Name of the layer to extract activations from.
        use_cache : bool
            If True, saves/loads results from disk to avoid recomputing.
            Default is True.

        Returns
        -------
        torch.Tensor
            Pooled activations of shape [N, C] where N is the number of
            random images and C is the number of channels at that layer.
        """
        cache_path = os.path.join(
            self.cache_dir,
            f"random_{self.model_wrapper.model_name}_{layer_name}.joblib"
        )

        # Try loading from cache first
        if use_cache and os.path.exists(cache_path):
            print(f"  Loading random activations from cache: {cache_path}")
            return load(cache_path)

        if self.random_dir is None:
            raise ValueError(
                "Random images directory not set. "
                "Call set_concepts() with random_dir parameter."
            )

        print(f"  Computing random activations at layer '{layer_name}'...")

        # Get feature maps for all random images
        feature_maps = self.model_wrapper.get_feature_maps_for_concept(
            self.random_dir, layer_name
        )

        # Global Average Pooling: average over spatial dimensions (H, W)
        # Shape: [N, C, H, W] → [N, C]
        pooled = F.adaptive_avg_pool2d(feature_maps, (1, 1))
        pooled = pooled.squeeze(-1).squeeze(-1)  # remove H and W dimensions

        if use_cache:
            dump(pooled, cache_path)
            print(f"  Random activations saved to cache.")

        return pooled

    def _compute_cavs(
        self,
        layer_name: str,
        concept_name: str,
        random_activations: torch.Tensor,
        use_cache: bool = True,
    ) -> Cav:
        """
        Compute the Concept Activation Vector (CAV) for a concept at a layer.

        The CAV is computed as:
        1. Extract feature maps for all concept images at this layer
        2. Pool each feature map to a vector (Global Average Pooling)
        3. Compute the positive centroid (mean of concept vectors)
        4. Compute the negative centroid (mean of random vectors)
        5. CAV direction = positive centroid - negative centroid
        6. Compute concept emblem (scale factor) using contraharmonic mean

        Parameters
        ----------
        layer_name : str
            Name of the layer to compute the CAV at.
        concept_name : str
            Name of the concept (e.g. "striped").
        random_activations : torch.Tensor
            Pre-computed pooled activations of random images. Shape: [N, C].
        use_cache : bool
            If True, saves/loads the CAV from disk. Default is True.

        Returns
        -------
        Cav
            The computed CAV object containing direction, centroids,
            and concept emblem.
        """
        cache_path = os.path.join(
            self.cache_dir,
            f"cav_{self.model_wrapper.model_name}_{layer_name}_{concept_name}.joblib"
        )

        # Try loading from cache first
        if use_cache and os.path.exists(cache_path):
            print(f"  Loading CAV from cache: {cache_path}")
            cav = load(cache_path)
            return cav.to(self.device)

        print(f"  Computing CAV for '{concept_name}' at layer '{layer_name}'...")

        concept_dir = self.concept_dirs[concept_name]

        # Step 1: Extract feature maps for concept images
        feature_maps = self.model_wrapper.get_feature_maps_for_concept(
            concept_dir, layer_name
        )
        # feature_maps shape: [N, C, H, W]

        # Step 2: Pool feature maps → one vector per image
        # Shape: [N, C, H, W] → [N, C]
        pooled_concepts = F.adaptive_avg_pool2d(feature_maps, (1, 1))
        pooled_concepts = pooled_concepts.squeeze(-1).squeeze(-1)

        # Step 3: Compute positive centroid (mean across all concept images)
        # Shape: [C]
        positive_centroid = torch.mean(pooled_concepts, dim=0)

        # Step 4: Compute negative centroid (mean across all random images)
        # Shape: [C]
        negative_centroid = torch.mean(random_activations, dim=0)

        # Step 5: CAV direction = positive - negative
        # This vector points "toward the concept" in the CNN's internal space
        direction = positive_centroid - negative_centroid

        # Step 6: Compute concept emblem (scale factor for concept map normalization)
        # Uses contraharmonic mean on the concept feature maps
        # This gives a reference scale for how strongly this concept activates
        concept_emblem = contraharmonic_mean(
            F.relu(feature_maps), axis=(2, 3)
        )
        concept_emblem = torch.mean(concept_emblem, dim=0)

        # Build and return the CAV object
        cav = Cav(
            concept_centroid=positive_centroid,
            negative_centroid=negative_centroid,
            direction=direction,
            concept_emblem=concept_emblem,
        )

        if use_cache:
            dump(cav.cpu(), cache_path)
            print(f"  CAV saved to cache.")

        return cav.to(self.device)

    # -----------------------------------------------------------------------
    # Integrated Gradients
    # -----------------------------------------------------------------------

    def _interpolate_feature_maps(
        self,
        feature_maps: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create M interpolated versions of the feature maps between a baseline
        and the actual feature maps.

        This is the first step of Integrated Gradients. Instead of computing
        the gradient at one point, IG computes it at many points along a
        straight line from the baseline (zeros) to the actual feature maps,
        then averages them.

        Think of it like this: instead of asking "how steep is the hill here?",
        you walk from the bottom to the top and average the steepness at each
        step.

        Parameters
        ----------
        feature_maps : torch.Tensor
            Actual feature maps from the test image. Shape: [1, C, H, W].
        baseline : torch.Tensor
            Baseline feature maps (usually zeros). Shape: [1, C, H, W].

        Returns
        -------
        torch.Tensor
            Interpolated feature maps. Shape: [m_steps, C, H, W].
            Each slice along dim=0 is one step between baseline and actual.
        """
        # Create m_steps evenly spaced values between 0 and 1
        # Shape: [m_steps, 1, 1, 1] for broadcasting
        alphas = torch.linspace(0, 1, self.m_steps).view(
            self.m_steps, 1, 1, 1
        )

        # Interpolate: baseline + alpha * (feature_maps - baseline)
        # When alpha=0: result = baseline
        # When alpha=1: result = feature_maps
        delta = feature_maps - baseline
        interpolated = baseline + alphas * delta

        return interpolated

    def _compute_integrated_gradients(
        self,
        feature_maps: torch.Tensor,
        layer_name: str,
        class_index: int,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients (IG) of a class score with respect
        to the feature maps.

        IG tells us how much each individual value in the feature maps
        contributed to the final class prediction. Values with high IG
        are the ones that "mattered most" for the prediction.

        The formula is:
        IG = (feature_maps - baseline) * mean(gradients along interpolation)

        Parameters
        ----------
        feature_maps : torch.Tensor
            Feature maps of the test image at a specific layer.
            Shape: [1, C, H, W].
        layer_name : str
            Name of the layer the feature maps come from.
        class_index : int
            Index of the class to explain (e.g. 340 for zebra).

        Returns
        -------
        torch.Tensor
            Integrated gradients. Shape: [C, H, W].
            Same shape as the input feature maps (without batch dimension).
        """
        # Baseline: all zeros (represents "no information")
        baseline = torch.zeros_like(feature_maps)

        # Step 1: Create interpolated feature maps between baseline and actual
        # Shape: [m_steps, C, H, W]
        interpolated = self._interpolate_feature_maps(feature_maps, baseline)

        # Step 2: Compute gradients at each interpolation step
        # Shape: [m_steps, C, H, W]
        gradients = self.model_wrapper.get_gradient_of_score(
            interpolated, layer_name, class_index
        )

        # Step 3: Average the gradients across all interpolation steps
        # Shape: [C, H, W]
        avg_gradients = torch.mean(gradients, dim=0)

        # Step 4: Multiply by (feature_maps - baseline)
        # This scales the gradients by how much each value actually changed
        # Shape: [C, H, W]
        integrated_gradients = (
            feature_maps.squeeze(0) - baseline.squeeze(0)
        ) * avg_gradients

        return integrated_gradients

    # -----------------------------------------------------------------------
    # Concept map computation
    # -----------------------------------------------------------------------

    def _compute_concept_map(
        self,
        feature_maps: torch.Tensor,
        cav: Cav,
    ) -> torch.Tensor:
        """
        Compute the raw concept map for a test image.

        The concept map is a spatial heatmap showing WHERE in the image
        the CNN has detected the presence of a concept.

        It is computed as a weighted sum of the feature maps, where the
        weights come from the CAV direction (similar to GradCAM):

        concept_map = ReLU( sum_k( cav_direction[k] * feature_maps[k] ) )

        Where k iterates over the C channels of the feature maps.

        Parameters
        ----------
        feature_maps : torch.Tensor
            Feature maps of the test image. Shape: [1, C, H, W].
        cav : Cav
            The CAV for the concept being analyzed.

        Returns
        -------
        torch.Tensor
            Raw concept map. Shape: [H, W].
            Values are non-negative (ReLU applied).
        """
        # CAV direction shape: [C]
        # We need it as [C, 1, 1] to multiply with feature maps [1, C, H, W]
        direction = cav.direction.to(self.device)
        direction = direction.view(-1, 1, 1)

        # Weighted sum across channels: [1, C, H, W] → [1, H, W]
        weighted = feature_maps.squeeze(0) * direction
        raw_map = weighted.sum(dim=0)

        # ReLU: keep only positive activations
        # (negative means the concept is ABSENT, we only care where it IS)
        raw_map = F.relu(raw_map)

        return raw_map

    def _normalize_concept_map(
        self,
        raw_map: torch.Tensor,
        cav: Cav,
    ) -> torch.Tensor:
        """
        Normalize a raw concept map using the concept emblem (scale factor).

        Without normalization, concept maps from different concepts or
        different images cannot be compared because they have different
        absolute scales.

        The normalization formula is:
        normalized_map[i,j] = min(1, raw_map[i,j] / (concept_emblem + epsilon))

        This clips values to [0, 1], making them interpretable and comparable.

        Parameters
        ----------
        raw_map : torch.Tensor
            Raw concept map. Shape: [H, W].
        cav : Cav
            The CAV containing the concept emblem (scale factor).

        Returns
        -------
        torch.Tensor
            Normalized concept map with values in [0, 1]. Shape: [H, W].
        """
        concept_emblem = cav.concept_emblem.to(self.device)

        # The concept emblem has shape [C] — take its mean as the scale factor
        scale = torch.mean(concept_emblem) + 1e-10

        # Normalize and clip to [0, 1]
        normalized = raw_map / scale
        normalized = torch.clamp(normalized, 0.0, 1.0)

        return normalized

    # -----------------------------------------------------------------------
    # Attribution score computation
    # -----------------------------------------------------------------------

    def _compute_attribution(
        self,
        feature_maps: torch.Tensor,
        concept_map: torch.Tensor,
        cav: Cav,
        layer_name: str,
        class_index: int,
    ) -> torch.Tensor:
        """
        Compute the attribution score for a concept at a layer for a class.

        The attribution score is a single number that measures HOW MUCH
        the concept contributed to the classification of the test image
        as a specific class.

        It is computed in three steps:
        1. Compute Integrated Gradients on the feature maps
           (how much does each feature map value matter for this class?)
        2. Multiply IG by the concept map
           (mask: keep only the parts where the concept IS present)
        3. Compute the dot product with the normalized CAV direction
           (project onto the concept direction to get a scalar score)

        Parameters
        ----------
        feature_maps : torch.Tensor
            Feature maps of the test image. Shape: [1, C, H, W].
        concept_map : torch.Tensor
            Normalized concept map. Shape: [H, W].
        cav : Cav
            The CAV for this concept.
        layer_name : str
            Name of the layer.
        class_index : int
            Index of the class to compute the attribution for.

        Returns
        -------
        torch.Tensor
            A single scalar tensor representing the attribution score.
            Higher values mean the concept contributed more to the prediction.
        """
        # Step 1: Compute Integrated Gradients
        # Shape: [C, H, W]
        ig = self._compute_integrated_gradients(
            feature_maps, layer_name, class_index
        )

        # Step 2: Multiply IG by concept map (element-wise)
        # concept_map shape: [H, W] → broadcast to [C, H, W]
        masked_ig = ig * concept_map.unsqueeze(0)

        # Apply ReLU: keep only positive contributions
        masked_ig = F.relu(masked_ig)

        # Step 3: Dot product with normalized CAV direction
        # direction shape: [C] → [C, 1, 1]
        direction = cav.direction.to(self.device)
        direction_norm = direction / (torch.norm(direction) + 1e-10)
        direction_norm = direction_norm.view(-1, 1, 1)

        # Element-wise multiply then sum → scalar
        attribution = (masked_ig * direction_norm).sum()

        return attribution

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(self, image_tensor: torch.Tensor, image_path: str) -> Predictions:
        """
        Run the model on an image and return the top predicted classes.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Preprocessed image tensor. Shape: [1, C, H, W].
        image_path : str
            Path to the image file (used for display).

        Returns
        -------
        Predictions
            Object containing top-k class predictions with names and
            confidence scores. Call .info() to print a formatted table.
        """
        # Get softmax probabilities for all classes
        probs = self.model_wrapper.get_predictions(image_tensor)
        # probs shape: [1, num_classes]

        probs_np = probs[0].numpy()

        # Build list of top n_classes predictions
        predictions = []
        for rank in range(1, self.n_classes + 1):
            idx = nth_highest_index(probs_np, rank)
            predictions.append(
                Prediction(
                    class_name=self.model_wrapper.id_to_label(idx),
                    class_index=idx,
                    confidence=float(probs_np[idx]),
                )
            )

        result = Predictions(
            predictions=[predictions],
            test_image_path=image_path,
            model_name=self.model_wrapper.model_name,
        )

        # Store target classes for use in explain()
        self.target_classes = [p.class_index for p in predictions]

        return result