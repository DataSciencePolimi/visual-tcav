"""
visual_tcav.py
--------------
Base class for Visual-TCAV. Contains all shared logic for computing
CAVs, integrated gradients, concept maps, and attribution scores.

Both LocalVisualTCAV and GlobalVisualTCAV inherit from this class.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import dump, load
from typing import Callable, Optional

from visual_tcav.utils import (
    Cav,
    ConceptLayer,
    Prediction,
    Predictions,
    contraharmonic_mean,
    nth_highest_index,
)

sys.dont_write_bytecode = True


def _load_model_wrapper(model, model_name, model_wrapper):
    """
    Resolve the model wrapper from the three supported input styles.

    Supports:
    - Style 1: model="resnet50" (string, auto-loads from torchvision)
    - Style 2: model=my_resnet (nn.Module, wrapped automatically)
    - Style 3: model_wrapper=my_wrapper (TorchModelWrapper, used directly)

    Parameters
    ----------
    model : str or nn.Module or None
        Model name string or PyTorch model object.
    model_name : str or None
        Optional display name for the model.
    model_wrapper : TorchModelWrapper or None
        Pre-built wrapper object.

    Returns
    -------
    TorchModelWrapper
        Ready-to-use model wrapper.
    """
    from visual_tcav.model_wrapper import TorchModelWrapper

    if model_wrapper is not None:
        return model_wrapper

    if isinstance(model, str):
        import torchvision.models as tv_models
        supported = {
            "resnet18":  tv_models.resnet18,
            "resnet50":  tv_models.resnet50,
            "resnet101": tv_models.resnet101,
            "vgg16":     tv_models.vgg16,
            "vgg19":     tv_models.vgg19,
        }
        if model not in supported:
            raise ValueError(
                f"Model string '{model}' not supported. "
                f"Supported names: {list(supported.keys())}. "
                f"For other models pass an nn.Module directly."
            )
        loaded = supported[model](weights="DEFAULT")
        return TorchModelWrapper(
            model_name=model_name or model,
            model=loaded,
        )

    if isinstance(model, nn.Module):
        return TorchModelWrapper(
            model_name=model_name or "model",
            model=model,
        )

    raise ValueError(
        "Provide one of:\n"
        "  model='resnet50'          (string — auto-loaded)\n"
        "  model=my_pytorch_model    (nn.Module — wrapped automatically)\n"
        "  model_wrapper=my_wrapper  (TorchModelWrapper — used directly)"
    )


class VisualTCAV:
    """
    Base class for Visual-TCAV.

    Implements the core pipeline shared by LocalVisualTCAV and GlobalVisualTCAV.
    Not meant to be used directly. Use LocalVisualTCAV or GlobalVisualTCAV.

    The model can be provided in three ways (in order of simplicity):

    1. As a string — auto-loaded from torchvision:
       ``LocalVisualTCAV(model="resnet50", ...)``

    2. As a PyTorch nn.Module — wrapped automatically:
       ``LocalVisualTCAV(model=my_resnet, ...)``

    3. As an explicit TorchModelWrapper — for full control:
       ``LocalVisualTCAV(model_wrapper=my_wrapper, ...)``

    Parameters
    ----------
    model : str or nn.Module, optional
        Model name (e.g. "resnet50") or PyTorch model object.
    model_name : str, optional
        Display name used for caching and logging. Inferred from model
        if not provided.
    model_wrapper : TorchModelWrapper, optional
        Pre-built wrapper for advanced use cases (custom preprocessing,
        custom labels, etc.).
    n_classes : int, optional
        Number of top predicted classes to explain. Default is 3.
    m_steps : int, optional
        Interpolation steps for Integrated Gradients. Default is 50.
    max_examples : int, optional
        Maximum number of concept/random images to use. Default is 500.
    cache_dir : str, optional
        Directory for caching CAVs and activations. Default is ".cache".
    cav_fn : callable, optional
        Custom CAV computation function. Must have signature:
        ``cav_fn(concept_features: Tensor, random_features: Tensor) -> Cav``
        If not provided, the default centroid difference method is used.
    """

    def __init__(
        self,
        model=None,
        model_name: str = None,
        model_wrapper=None,
        n_classes: int = 3,
        m_steps: int = 50,
        max_examples: int = 500,
        cache_dir: str = ".cache",
        cav_fn: Optional[Callable] = None,
    ):
        self.model_wrapper = _load_model_wrapper(model, model_name, model_wrapper)
        self.n_classes = n_classes
        self.m_steps = m_steps
        self.max_examples = max_examples
        self.cache_dir = cache_dir

        self.concept_names = []
        self.concept_dirs = {}
        self.layer_names = []
        self.random_dir = None
        self.target_classes = []

        # Main storage: computations[layer_name][concept_name] = ConceptLayer
        self.computations = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use provided CAV function or fall back to centroid difference default
        self.cav_fn = cav_fn if cav_fn is not None else self._default_cav_fn

        os.makedirs(self.cache_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Setup
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

        Parameters
        ----------
        concept_names : list of str
            Names of the concepts (e.g. ["striped", "dotted"]).
        concept_dirs : dict, optional
            Explicit mapping of concept name to image folder path.
        concept_base_dir : str, optional
            Base folder containing one subfolder per concept.
        random_dir : str, optional
            Folder containing random (negative) images.

        Raises
        ------
        ValueError
            If neither concept_dirs nor concept_base_dir is provided,
            or if any concept folder does not exist.
        """
        if concept_dirs is not None:
            self.concept_dirs = concept_dirs
        elif concept_base_dir is not None:
            self.concept_dirs = {
                name: os.path.join(concept_base_dir, name)
                for name in concept_names
            }
        else:
            raise ValueError(
                "Provide either 'concept_dirs' or 'concept_base_dir'."
            )

        for name, path in self.concept_dirs.items():
            if not os.path.exists(path):
                raise ValueError(
                    f"Concept folder for '{name}' not found at: {path}"
                )

        self.concept_names = concept_names

        if random_dir is not None:
            self.random_dir = random_dir
        elif concept_base_dir is not None:
            self.random_dir = os.path.join(concept_base_dir, "random")

        if self.random_dir and not os.path.exists(self.random_dir):
            raise ValueError(
                f"Random images folder not found at: {self.random_dir}"
            )

        print(f"Concepts set: {self.concept_names}")

    def set_layers(self, layer_names: list) -> None:
        """
        Specify which CNN layers to analyze.

        Parameters
        ----------
        layer_names : list of str
            Layer names (e.g. ["layer3", "layer4"]).
            Call model_wrapper.info() to see available layers.

        Raises
        ------
        ValueError
            If no layer names are provided.
        """
        if not layer_names:
            raise ValueError(
                "Provide at least one layer name. "
                "Call model_wrapper.info() to see available layers."
            )
        self.layer_names = layer_names
        self.computations = {
            layer: {concept: ConceptLayer() for concept in self.concept_names}
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
        Compute (or load) pooled activations of random images at a layer.

        Random images serve as negative examples for CAV computation.
        Global Average Pooling reduces [N, C, H, W] to [N, C] so that
        each image is represented as a single vector.

        Parameters
        ----------
        layer_name : str
            Layer to extract activations from.
        use_cache : bool
            Save/load results from disk. Default is True.

        Returns
        -------
        torch.Tensor
            Pooled activations of shape [N, C].
        """
        cache_path = os.path.join(
            self.cache_dir,
            f"random_{self.model_wrapper.model_name}_{layer_name}.joblib"
        )

        if use_cache and os.path.exists(cache_path):
            print(f"  Loading random activations from cache.")
            return load(cache_path)

        if self.random_dir is None:
            raise ValueError(
                "Random images directory not set. "
                "Call set_concepts() with the random_dir parameter."
            )

        print(f"  Computing random activations at '{layer_name}'...")

        feature_maps = self.model_wrapper.get_feature_maps_for_concept(
            self.random_dir, layer_name
        )

        # GAP reduces spatial dimensions: [N, C, H, W] -> [N, C]
        pooled = F.adaptive_avg_pool2d(feature_maps, (1, 1))
        pooled = pooled.squeeze(-1).squeeze(-1)

        if use_cache:
            dump(pooled, cache_path)

        return pooled

    def _default_cav_fn(
        self,
        concept_features: torch.Tensor,
        random_features: torch.Tensor,
    ) -> Cav:
        """
        Default CAV computation using centroid difference.

        Computes the direction as the difference between the mean concept
        activation (positive centroid) and the mean random activation
        (negative centroid), as described in the Visual-TCAV paper.

        Parameters
        ----------
        concept_features : torch.Tensor
            Pooled concept image activations. Shape: [N, C].
        random_features : torch.Tensor
            Pooled random image activations. Shape: [N, C].

        Returns
        -------
        Cav
            CAV with direction, centroids, and concept emblem set.
        """
        positive_centroid = torch.mean(concept_features, dim=0)
        negative_centroid = torch.mean(random_features, dim=0)

        # Direction points from "random" toward "concept" in feature space
        direction = positive_centroid - negative_centroid

        # Concept emblem: scale factor for concept map normalization
        concept_emblem = contraharmonic_mean(
            F.relu(concept_features.unsqueeze(-1).unsqueeze(-1)), axis=(2, 3)
        )
        concept_emblem = torch.mean(concept_emblem, dim=0)

        return Cav(
            concept_centroid=positive_centroid,
            negative_centroid=negative_centroid,
            direction=direction,
            concept_emblem=concept_emblem,
        )

    def _compute_cavs(
        self,
        layer_name: str,
        concept_name: str,
        random_activations: torch.Tensor,
        use_cache: bool = True,
    ) -> Cav:
        """
        Compute (or load) the CAV for a concept at a specific layer.

        Handles caching and feature extraction, then delegates the actual
        CAV computation to self.cav_fn (default or user-provided).

        Parameters
        ----------
        layer_name : str
            Layer to compute the CAV at.
        concept_name : str
            Name of the concept (e.g. "striped").
        random_activations : torch.Tensor
            Pre-computed pooled activations of random images. Shape: [N, C].
        use_cache : bool
            Save/load the CAV from disk. Default is True.

        Returns
        -------
        Cav
            The computed CAV.
        """
        cache_path = os.path.join(
            self.cache_dir,
            f"cav_{self.model_wrapper.model_name}_{layer_name}_{concept_name}.joblib"
        )

        if use_cache and os.path.exists(cache_path):
            print(f"  Loading CAV from cache: {concept_name} @ {layer_name}")
            cav = load(cache_path)
            return cav.to(self.device)

        print(f"  Computing CAV: '{concept_name}' @ '{layer_name}'...")

        feature_maps = self.model_wrapper.get_feature_maps_for_concept(
            self.concept_dirs[concept_name], layer_name
        )

        # GAP: [N, C, H, W] -> [N, C]
        pooled_concepts = F.adaptive_avg_pool2d(feature_maps, (1, 1))
        pooled_concepts = pooled_concepts.squeeze(-1).squeeze(-1)

        # Delegate to cav_fn — default centroid method or user-provided function
        cav = self.cav_fn(pooled_concepts, random_activations)

        if use_cache:
            dump(cav.cpu(), cache_path)

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
        Create m_steps interpolations between baseline and actual feature maps.

        Integrated Gradients averages gradients along a straight path from
        a zero baseline to the actual feature maps, rather than at a single point.

        Parameters
        ----------
        feature_maps : torch.Tensor
            Actual feature maps. Shape: [1, C, H, W].
        baseline : torch.Tensor
            Baseline (typically zeros). Shape: [1, C, H, W].

        Returns
        -------
        torch.Tensor
            Interpolated feature maps. Shape: [m_steps, C, H, W].
        """
        alphas = torch.linspace(0, 1, self.m_steps).view(self.m_steps, 1, 1, 1)
        delta = feature_maps - baseline
        return baseline + alphas * delta

    def _compute_integrated_gradients(
        self,
        feature_maps: torch.Tensor,
        layer_name: str,
        class_index: int,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients of a class score w.r.t. feature maps.

        Parameters
        ----------
        feature_maps : torch.Tensor
            Feature maps of the test image. Shape: [1, C, H, W].
        layer_name : str
            Layer the feature maps come from.
        class_index : int
            Index of the class to explain.

        Returns
        -------
        torch.Tensor
            Integrated gradients. Shape: [C, H, W].
        """
        baseline = torch.zeros_like(feature_maps)
        interpolated = self._interpolate_feature_maps(feature_maps, baseline)

        gradients = self.model_wrapper.get_gradient_of_score(
            interpolated, layer_name, class_index
        )

        avg_gradients = torch.mean(gradients, dim=0)

        # Scale by the actual input change from baseline to feature maps
        return (feature_maps.squeeze(0) - baseline.squeeze(0)) * avg_gradients

    # -----------------------------------------------------------------------
    # Concept map
    # -----------------------------------------------------------------------

    def _compute_concept_map(
        self,
        feature_maps: torch.Tensor,
        cav: Cav,
    ) -> torch.Tensor:
        """
        Compute the raw concept map for a test image.

        Analogous to GradCAM: weighted sum of feature maps using CAV direction
        as weights, then ReLU to keep only locations where concept is present.

        Parameters
        ----------
        feature_maps : torch.Tensor
            Feature maps of the test image. Shape: [1, C, H, W].
        cav : Cav
            The CAV for the concept being analyzed.

        Returns
        -------
        torch.Tensor
            Raw concept map. Shape: [H, W]. Non-negative values only.
        """
        # Reshape direction [C] -> [C, 1, 1] for broadcasting with [C, H, W]
        direction = cav.direction.to(self.device).view(-1, 1, 1)
        weighted = feature_maps.squeeze(0) * direction
        raw_map = weighted.sum(dim=0)
        # Negative values indicate the concept is absent — discard them
        return F.relu(raw_map)

    def _normalize_concept_map(
        self,
        raw_map: torch.Tensor,
        cav: Cav,
    ) -> torch.Tensor:
        """
        Normalize a raw concept map to [0, 1] using the concept emblem.

        Without normalization, concept maps from different concepts or images
        are not comparable due to differing absolute activation scales.

        Parameters
        ----------
        raw_map : torch.Tensor
            Raw concept map. Shape: [H, W].
        cav : Cav
            The CAV containing the concept emblem (scale factor).

        Returns
        -------
        torch.Tensor
            Normalized concept map with values in [0, 1].
        """
        concept_emblem = cav.concept_emblem.to(self.device)
        scale = torch.mean(concept_emblem) + 1e-10
        return torch.clamp(raw_map / scale, 0.0, 1.0)

    # -----------------------------------------------------------------------
    # Attribution score
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

        Combines IG with the concept map: IG identifies which feature map
        values matter for the class, the concept map masks to regions where
        the concept is present, and dot product with CAV yields a scalar.

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
            Index of the class to compute attribution for.

        Returns
        -------
        torch.Tensor
            Scalar attribution score.
        """
        ig = self._compute_integrated_gradients(
            feature_maps, layer_name, class_index
        )

        # Mask IG with concept map: keep only regions where concept is present
        masked_ig = F.relu(ig * concept_map.unsqueeze(0))

        direction = cav.direction.to(self.device)
        direction_norm = (direction / (torch.norm(direction) + 1e-10)).view(-1, 1, 1)

        return (masked_ig * direction_norm).sum()

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
            Path to the image file (used for display purposes).

        Returns
        -------
        Predictions
            Top-k class predictions. Call .info() to print a table.
        """
        probs = self.model_wrapper.get_predictions(image_tensor)
        probs_np = probs[0].numpy()

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

        self.target_classes = [p.class_index for p in predictions]
        return result