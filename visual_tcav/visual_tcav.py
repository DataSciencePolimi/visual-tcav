"""
visual_tcav.py
--------------
Base class for Visual-TCAV. Contains all shared logic for computing
CAVs, integrated gradients, concept maps, and attribution scores.

Both LocalVisualTCAV and GlobalVisualTCAV inherit from this class.
"""

import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import dump, load
from typing import Callable, Optional

from .utils import (
    Cav,
    ConceptLayer,
    Prediction,
    Predictions,
    contraharmonic_mean,
    nth_highest_index,
)

sys.dont_write_bytecode = True


# Supported torchvision models for string-based loading
_SUPPORTED_MODELS = {
    "resnet18":  ("torchvision.models", "resnet18",  "ResNet18_Weights"),
    "resnet50":  ("torchvision.models", "resnet50",  "ResNet50_Weights"),
    "resnet101": ("torchvision.models", "resnet101", "ResNet101_Weights"),
    "vgg16":     ("torchvision.models", "vgg16",     "VGG16_Weights"),
    "vgg19":     ("torchvision.models", "vgg19",     "VGG19_Weights"),
}


def available_layers(model, model_name: str = None) -> None:
    """
    Print the available CNN layers for a model.

    Use this utility function before instantiating LocalVisualTCAV or
    GlobalVisualTCAV to decide which layers to analyze. Deeper layers
    (closer to the output) capture higher-level concepts.

    Parameters
    ----------
    model : str or nn.Module
        Model name string (e.g. "resnet50") or a PyTorch model object.
    model_name : str, optional
        Display name for the model. Required when passing an nn.Module
        of a known torchvision model to enable label auto-loading
        (e.g. model_name="resnet50").

    Examples
    --------
    >>> from visual_tcav import available_layers
    >>>
    >>> # From a model name string
    >>> available_layers("resnet50")
    +----------------------------+
    |       Model: resnet50      |
    +------------+---------------+
    | N. classes |    Layers     |
    +------------+---------------+
    |    1000    |    layer1     |
    |            |    layer2     |
    |            |    layer3     |
    |            |    layer4     |
    +------------+---------------+
    >>>
    >>> # From an nn.Module
    >>> import torchvision.models as models
    >>> my_resnet = models.resnet50(weights='DEFAULT')
    >>> available_layers(my_resnet, model_name="resnet50")
    """
    from prettytable import PrettyTable

    if isinstance(model, str):
        # String path — full wrapper with labels available
        wrapper = _build_wrapper(model, model_name=model_name)
        wrapper.info()
        return

    if isinstance(model, nn.Module):
        # Try full wrapper first (labels available for known models)
        try:
            wrapper = _build_wrapper(model, model_name=model_name)
            wrapper.info()
        except ValueError:
            # Unknown model — show layer names only, no labels needed
            name = model_name or model.__class__.__name__
            layer_names = [
                n for n, m in model.named_children()
                if not isinstance(m, (nn.Linear, nn.Flatten))
                and n not in ("avgpool", "fc", "classifier")
            ]
            table = PrettyTable(
                title=f"Model: {name}",
                field_names=["Layers"],
            )
            for layer in layer_names:
                table.add_row([layer])
            print(table)
        return

    raise ValueError(
        "model must be a string (e.g. 'resnet50') or a PyTorch nn.Module."
    )


def _build_wrapper(model, model_name):
    """
    Build a TorchModelWrapper from a model string or nn.Module.

    Parameters
    ----------
    model : str or nn.Module
        Model name string (e.g. "resnet50") or PyTorch model object.
    model_name : str or None
        Optional display name. Inferred from model string if not given.

    Returns
    -------
    TorchModelWrapper
        Ready-to-use wrapper with labels and preprocessing resolved.
    """
    import torchvision.models as tv
    from .model_wrapper import TorchModelWrapper

    if isinstance(model, str):
        name = model.lower()
        if name not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model}' not supported for string loading.\n"
                f"Supported: {list(_SUPPORTED_MODELS.keys())}.\n"
                f"For other models, pass an nn.Module directly."
            )
        fn_name, weights_cls = _SUPPORTED_MODELS[name][1], _SUPPORTED_MODELS[name][2]
        fn = getattr(tv, fn_name)
        weights = getattr(tv, weights_cls).DEFAULT
        loaded_model = fn(weights=weights)
        return TorchModelWrapper(
            model_name=model_name or name,
            model=loaded_model,
            labels=list(weights.meta["categories"]),
            model_preprocess=weights.transforms(),
        )

    if isinstance(model, nn.Module):
        _weights_obj = None

        if (
            hasattr(model, "_weights")
            and model._weights is not None
            and hasattr(model._weights, "meta")
            and "categories" in model._weights.meta
        ):
            _weights_obj = model._weights

        if _weights_obj is None and model_name is not None:
            _name = model_name.lower()
            if _name in _SUPPORTED_MODELS:
                weights_cls = _SUPPORTED_MODELS[_name][2]
                _weights_obj = getattr(tv, weights_cls).DEFAULT

        if _weights_obj is not None:
            return TorchModelWrapper(
                model_name=model_name or "model",
                model=model,
                labels=list(_weights_obj.meta["categories"]),
                model_preprocess=_weights_obj.transforms(),
            )
        else:
            raise ValueError(
                f"Could not auto-load labels for this model.\n"
                f"For standard torchvision models, pass model_name as one of: "
                f"{list(_SUPPORTED_MODELS.keys())}.\n"
                f"For custom models, use TorchModelWrapper directly:\n"
                f"  wrapper = TorchModelWrapper(model_name='my_model', "
                f"model=my_model, labels=[...])\n"
                f"  tcav = LocalVisualTCAV(model_wrapper=wrapper, ...)"
            )

    raise ValueError(
        "Provide either:\n"
        "  model='resnet50'       (string — auto-loaded)\n"
        "  model=my_pytorch_model (nn.Module — wrapped automatically)\n"
        "  model_wrapper=wrapper  (TorchModelWrapper — advanced use)"
    )


class VisualTCAV:
    """
    Base class for Visual-TCAV.

    Implements the core pipeline shared by LocalVisualTCAV and GlobalVisualTCAV.
    Not meant to be used directly — use LocalVisualTCAV or GlobalVisualTCAV.

    Before instantiating, use the standalone utility to inspect available layers:

    .. code-block:: python

        from visual_tcav import available_layers
        available_layers("resnet50")

    The model is provided via one of two standard interfaces:

    **String** — auto-loads the model with default ImageNet weights:

    .. code-block:: python

        tcav = LocalVisualTCAV(model="resnet50", ...)

    **nn.Module** — use your own model (fine-tuned, custom weights, etc.):

    .. code-block:: python

        my_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        tcav = LocalVisualTCAV(model=my_model, model_name="resnet50", ...)

    Parameters
    ----------
    model : str or nn.Module, optional
        Model name string or PyTorch model object.
    model_name : str, optional
        Display name for the model.
    model_wrapper : TorchModelWrapper, optional
        Pre-built wrapper for advanced use cases.
    n_classes : int, optional
        Number of top predicted classes to explain. Default is 3.
    m_steps : int, optional
        Interpolation steps for Integrated Gradients. Default is 50.
    max_examples : int, optional
        Maximum number of concept/random images to use. Default is 500.
    cache_dir : str, optional
        Directory for caching CAVs and activations. Default is ".cache".
        Set to None to disable caching entirely.
    cav_fn : callable, optional
        Custom CAV computation function. Must accept two tensors of
        shape [N, C] and return a Cav object. If not provided, the
        default centroid difference method is used.
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
        if model_wrapper is not None:
            self.model_wrapper = model_wrapper
        elif model is not None:
            self.model_wrapper = _build_wrapper(model, model_name)
        else:
            raise ValueError(
                "Provide either 'model' (string or nn.Module) "
                "or 'model_wrapper' (TorchModelWrapper)."
            )

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

        # Full (non-pooled) feature maps of the random images, kept in memory
        # only while the current layer is being processed. They are needed
        # to compute the concept emblem (see _compute_emblem) and are never
        # written to disk. Only one layer at a time is held, because early
        # layers have large maps (hundreds of MB for a few hundred images).
        self._random_feature_maps = {}
        self._emblem_chunk_size = 32

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use provided CAV function or fall back to centroid difference default
        self.cav_fn = cav_fn if cav_fn is not None else self._default_cav_fn

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Cache management
    # -----------------------------------------------------------------------

    def clear_cache(self) -> None:
        """
        Delete all cached CAVs and random activations.

        Cached files are stored in ``cache_dir`` and named after the model,
        layer, and concept. Use this method to force full recomputation on
        the next call to explain(), or to free disk space.

        Uses ``shutil.rmtree`` to remove the cache directory and recreates
        it empty.

        Examples
        --------
        >>> tcav.clear_cache()
        Cache cleared: .cache
        """
        if self.cache_dir is None:
            print("Caching is disabled (cache_dir=None). Nothing to clear.")
            return
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Cache cleared: {self.cache_dir}")
        else:
            print(f"Cache directory does not exist: {self.cache_dir}")

    # -----------------------------------------------------------------------
    # Internal setup — called by subclass constructors only
    # -----------------------------------------------------------------------

    def _setup_concepts(
        self,
        concept_names: list,
        concept_dirs: dict = None,
        concept_base_dir: str = None,
        random_dir: str = None,
    ) -> None:
        """
        Configure concepts. Called internally by the constructor.

        Parameters
        ----------
        concept_names : list of str
            Names of the concepts (e.g. ["striped", "dotted"]).
        concept_dirs : dict, optional
            Explicit mapping of concept name to image folder path.
            Takes priority over concept_base_dir.
        concept_base_dir : str, optional
            Base folder containing one subfolder per concept.
        random_dir : str, optional
            Folder containing random images (reference distribution).

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
                "Provide either 'concept_dirs' (explicit path per concept) "
                "or 'concept_base_dir' (folder containing one subfolder per concept)."
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
                f"Random images folder not found at: {self.random_dir}\n"
                f"Pass random_dir explicitly or create a 'random' subfolder "
                f"inside concept_base_dir."
            )

        print(f"Concepts set: {self.concept_names}")

    def _setup_layers(self, layer_names: list) -> None:
        """
        Configure layers. Called internally by the constructor.

        Parameters
        ----------
        layer_names : list of str
            Layer names (e.g. ["layer3", "layer4"]).

        Raises
        ------
        ValueError
            If no layer names are provided.
        """
        if not layer_names:
            raise ValueError(
                "Provide at least one layer name.\n"
                "Use available_layers(model) to see valid layer names."
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
        self, layer_name: str, force_recompute: bool = False
    ) -> torch.Tensor:
        """
        Compute (or load) pooled activations of random images at a layer.

        Random images serve as the reference distribution for CAV computation.
        Global Average Pooling reduces [N, C, H, W] to [N, C] so that
        each image is represented as a single vector.

        Parameters
        ----------
        layer_name : str
            Layer to extract activations from.
        force_recompute : bool
            If True, ignores existing cache and recomputes. Default is False.

        Returns
        -------
        torch.Tensor
            Pooled activations of shape [N, C].
        """
        cache_path = (
            os.path.join(
                self.cache_dir,
                f"random_{self.model_wrapper.model_name}_{layer_name}.joblib"
            )
            if self.cache_dir is not None else None
        )

        if cache_path and not force_recompute and os.path.exists(cache_path):
            print(f"  Loading random activations from cache.")
            return load(cache_path)

        if self.random_dir is None:
            raise ValueError(
                "random_dir not set. Pass it to the constructor:\n"
                "  LocalVisualTCAV(..., random_dir='./path/to/random/', ...)"
            )

        print(f"  Computing random activations at '{layer_name}'...")

        feature_maps = self.model_wrapper.get_feature_maps_for_concept(
            self.random_dir, layer_name
        )
        self._random_feature_maps.clear()
        self._random_feature_maps[layer_name] = feature_maps

        # GAP reduces spatial dimensions: [N, C, H, W] -> [N, C]
        pooled = F.adaptive_avg_pool2d(feature_maps, (1, 1))
        pooled = pooled.squeeze(-1).squeeze(-1)

        if cache_path:
            dump(pooled, cache_path)

        return pooled

    def _get_random_feature_maps(self, layer_name: str) -> torch.Tensor:
        """
        Return the full [N, C, H, W] feature maps of the random images.

        Uses the maps kept in memory by _compute_random_activations() when
        available; otherwise (pooled activations were loaded from cache)
        runs one extra forward pass over the random folder.
        """
        if layer_name not in self._random_feature_maps:
            if self.random_dir is None:
                raise ValueError(
                    "random_dir not set. Pass it to the constructor:\n"
                    "  LocalVisualTCAV(..., random_dir='./path/to/random/', ...)"
                )
            self._random_feature_maps.clear()
            self._random_feature_maps[layer_name] = (
                self.model_wrapper.get_feature_maps_for_concept(
                    self.random_dir, layer_name
                )
            )
        return self._random_feature_maps[layer_name]

    def _release_random_feature_maps(self, layer_name: str) -> None:
        """
        Free the full random feature maps of a layer once its CAVs are done.

        Called by the explainers at the end of each layer. The pooled
        activations (and the cache on disk) are not affected.
        """
        self._random_feature_maps.pop(layer_name, None)

    def _default_cav_fn(
        self,
        concept_features: torch.Tensor,
        random_features: torch.Tensor,
    ) -> Cav:
        """
        Default CAV computation using centroid difference.

        Computes the direction as the difference between the mean concept
        activation (concept centroid) and the mean random image activation
        (random centroid), as described in the Visual-TCAV paper.

        Parameters
        ----------
        concept_features : torch.Tensor
            Pooled concept image activations. Shape: [N, C].
        random_features : torch.Tensor
            Pooled random image activations. Shape: [N, C].

        Returns
        -------
        Cav
            CAV with direction and centroids set. The concept emblem is
            not computed here: it needs the full spatial feature maps and
            is filled in by _compute_cavs() (see _compute_emblem).
        """
        concept_centroid = torch.mean(concept_features, dim=0)
        random_centroid = torch.mean(random_features, dim=0)

        # Direction points from random toward concept in feature space
        direction = concept_centroid - random_centroid

        return Cav(
            concept_centroid=concept_centroid,
            negative_centroid=random_centroid,
            direction=direction,
            concept_emblem=None,
        )

    def _compute_emblem(
        self,
        direction: torch.Tensor,
        concept_feature_maps: torch.Tensor,
        random_feature_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the concept emblem used to normalize concept maps.

        Follows the reference implementation: every concept (and random)
        image is projected on the CAV direction to obtain its raw concept
        map, the map is rectified and summarized by its contraharmonic
        mean over space, and the median over images is taken. The result
        is the pair (positive emblem, negative emblem): the typical map
        magnitude when the concept is present and when it is absent.

        Parameters
        ----------
        direction : torch.Tensor
            CAV direction. Shape: [C].
        concept_feature_maps : torch.Tensor
            Feature maps of the concept images. Shape: [N, C, H, W].
        random_feature_maps : torch.Tensor
            Feature maps of the random images. Shape: [M, C, H, W].

        Returns
        -------
        torch.Tensor
            Tensor of shape [2]: (positive_emblem, negative_emblem).
        """
        device = direction.device
        d = direction.view(1, -1, 1, 1)

        def _emblems(fmaps: torch.Tensor) -> torch.Tensor:
            # Images are projected in chunks: the projected maps of a few
            # hundred images at an early layer would otherwise need
            # several GB at once. Each image is independent, so the
            # result is the same as projecting all of them together.
            values = []
            for start in range(0, fmaps.shape[0], self._emblem_chunk_size):
                chunk = fmaps[start:start + self._emblem_chunk_size].to(device)
                projected = F.relu((d * chunk).sum(dim=1))              # [n, H, W]
                values.append(contraharmonic_mean(projected, axis=(1, 2)))  # [n]
            return torch.cat(values)

        positive = torch.quantile(_emblems(concept_feature_maps), 0.5)
        negative = torch.quantile(_emblems(random_feature_maps), 0.5)
        return torch.stack([positive, negative]).to(torch.float32)

    def _compute_cavs(
        self,
        layer_name: str,
        concept_name: str,
        random_activations: torch.Tensor,
        force_recompute: bool = False,
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
        force_recompute : bool
            If True, ignores existing cache and recomputes. Default is False.

        Returns
        -------
        Cav
            The computed CAV.
        """
        cache_path = (
            os.path.join(
                self.cache_dir,
                f"cav_{self.model_wrapper.model_name}_{layer_name}_{concept_name}.joblib"
            )
            if self.cache_dir is not None else None
        )

        if cache_path and not force_recompute and os.path.exists(cache_path):
            cav = load(cache_path)
            if self._has_valid_emblem(cav):
                print(f"  Loading CAV from cache: {concept_name} @ {layer_name}")
                return cav.to(self.device)
            # Cache written by an older version (scalar emblem): recompute
            print(f"  Stale CAV cache for {concept_name} @ {layer_name}, recomputing.")

        print(f"  Computing CAV: '{concept_name}' @ '{layer_name}'...")

        feature_maps = self.model_wrapper.get_feature_maps_for_concept(
            self.concept_dirs[concept_name], layer_name
        )

        # GAP: [N, C, H, W] -> [N, C]
        pooled_concepts = F.adaptive_avg_pool2d(feature_maps, (1, 1))
        pooled_concepts = pooled_concepts.squeeze(-1).squeeze(-1)

        # Delegate to cav_fn — default centroid method or user-provided function
        cav = self.cav_fn(pooled_concepts, random_activations)

        # The emblem needs the full spatial maps, so it is computed here
        # rather than inside cav_fn. A custom cav_fn may still provide its
        # own (positive, negative) pair, in which case it is kept.
        if not self._has_valid_emblem(cav):
            cav.concept_emblem = self._compute_emblem(
                cav.direction.to(self.device),
                feature_maps,
                self._get_random_feature_maps(layer_name),
            )

        if cache_path:
            dump(cav.cpu(), cache_path)

        return cav.to(self.device)

    @staticmethod
    def _has_valid_emblem(cav: Cav) -> bool:
        """True when the CAV carries a (positive, negative) emblem pair."""
        emblem = getattr(cav, "concept_emblem", None)
        return isinstance(emblem, torch.Tensor) and emblem.numel() == 2

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
            Interpolated feature maps. Shape: [m_steps + 1, C, H, W]
            (alpha = 0, 1/m, ..., 1, so that the trapezoidal rule can be
            applied over m_steps intervals, as in the reference).
        """
        alphas = torch.linspace(
            0, 1, self.m_steps + 1, device=feature_maps.device
        ).view(self.m_steps + 1, 1, 1, 1)
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

        # get_gradient_of_score() returns on CPU (it accumulates batches on the
        # compute device and transfers once). Bring the result back onto the
        # device the feature maps live on before combining the two, otherwise
        # this line raises a device mismatch on GPU.
        gradients = self.model_wrapper.get_gradient_of_score(
            interpolated, layer_name, class_index
        ).to(feature_maps.device)

        # Trapezoidal rule over the m_steps intervals (reference behaviour)
        avg_gradients = ((gradients[:-1] + gradients[1:]) / 2.0).mean(dim=0)

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
        # Values below zero indicate the concept is absent — discard them
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

        With the default emblem, a pair (positive, negative) computed as in
        the reference implementation, the map is clipped to
        [negative, positive] and rescaled linearly to [0, 1]. If a custom
        cav_fn provides no emblem, the map is divided by its own maximum.

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
        if self._has_valid_emblem(cav):
            positive, negative = cav.concept_emblem.to(self.device)
            if positive > negative:
                # Values below the negative emblem are background noise
                # (-> 0); values above the positive emblem are saturated (-> 1)
                clipped = torch.clamp(raw_map, min=negative, max=positive)
                return (clipped - negative) / (positive - negative)
            # The concept is not better recognized than random noise
            return torch.zeros_like(raw_map)

        # Fallback when a custom cav_fn provides no emblem
        scale = raw_map.max() + 1e-10
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
        values matter for the class, the concept map masks to the regions
        where the concept is present, and the dot product with the rectified
        CAV direction yields a scalar. Both operands of that dot product are
        non-negative, so the returned score is non-negative as well.

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
        fmaps = feature_maps.squeeze(0)                       # [C, H, W]

        # --- Integrated Gradients for this class ---------------------------
        ig = self._compute_integrated_gradients(
            feature_maps, layer_name, class_index
        )

        # --- Expected IG magnitude for this class --------------------------
        # The logit gap between the real feature maps and the zero baseline
        # sets the scale the attribution is normalized to, so that scores are
        # comparable across classes and images.
        with torch.no_grad():
            logits = self.model_wrapper.get_logits(feature_maps, layer_name)[0]
            logits_baseline = self.model_wrapper.get_logits(
                torch.zeros_like(feature_maps), layer_name
            )[0]

        ig_expected = F.relu(logits - logits_baseline)
        ig_expected_max = ig_expected.max()
        if ig_expected_max > 0:
            ig_expected = ig_expected / ig_expected_max
        ig_expected_class = ig_expected[class_index].to(fmaps.device)

        # --- Non-negative attribution tensor, rescaled ---------------------
        # ig already equals (feature_maps - baseline) * average gradient,
        # i.e. the reference's relu(gradient * feature_maps) once rectified.
        attributions = F.relu(ig)
        attributions = attributions * (
            ig_expected_class / (attributions.sum() + 1e-10)
        )

        # --- Mask with the concept map, then pool over space ---------------
        masked = attributions * concept_map.unsqueeze(0)
        pooled_masked = masked.sum(dim=(1, 2))                # [C], >= 0

        # --- Rectified, max-normalized CAV direction -----------------------
        # Rectifying the direction is what keeps the final score non-negative:
        # the dot product below is between two non-negative vectors. Layers
        # that can emit negative feature maps get a per-channel sign
        # correction first.
        direction = cav.direction.to(fmaps.device)
        if fmaps.min() < 0:
            sign = torch.where(
                (fmaps * concept_map.unsqueeze(0)).sum(dim=(1, 2)) < 0,
                -1.0,
                1.0,
            )
            pooled_cav_norm = F.relu(direction * sign)
        else:
            pooled_cav_norm = F.relu(direction)

        max_cav = pooled_cav_norm.max()
        if max_cav > 0:
            pooled_cav_norm = pooled_cav_norm / max_cav       # [C], in [0, 1]

        return torch.dot(pooled_cav_norm, pooled_masked)

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