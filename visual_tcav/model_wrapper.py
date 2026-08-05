"""
model_wrapper.py
----------------
Wrapper around any PyTorch CNN model, providing a consistent interface
for Visual-TCAV to extract predictions, feature maps, and gradients.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from prettytable import PrettyTable

sys.dont_write_bytecode = True


# Mapping of known torchvision model names to their default weights.
# Used to auto-load labels and preprocessing when no explicit values are given.
_TORCHVISION_WEIGHTS = None


def _get_torchvision_weights():
    """Lazy-load the torchvision weights map to avoid import overhead."""
    global _TORCHVISION_WEIGHTS
    if _TORCHVISION_WEIGHTS is None:
        import torchvision.models as tv
        _TORCHVISION_WEIGHTS = {
            "resnet18":  tv.ResNet18_Weights.DEFAULT,
            "resnet50":  tv.ResNet50_Weights.DEFAULT,
            "resnet101": tv.ResNet101_Weights.DEFAULT,
            "vgg16":     tv.VGG16_Weights.DEFAULT,
            "vgg19":     tv.VGG19_Weights.DEFAULT,
        }
    return _TORCHVISION_WEIGHTS


def _safe_batch_size(n_samples: int, desired: int = 32) -> int:
    """
    Find the largest batch size <= desired that evenly divides n_samples.

    Uneven batches cause errors during gradient computation, so we ensure
    every batch has exactly the same number of samples.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    desired : int
        Starting batch size to try. Default is 32.

    Returns
    -------
    int
        A batch size that evenly divides n_samples.
    """
    size = min(desired, n_samples)
    while size > 1 and n_samples % size != 0:
        size -= 1
    return size


class _FeatureMapsModel(nn.Module):
    """
    Runs the input through a CNN up to a specific layer and returns
    the feature maps (activations) at that point.

    Parameters
    ----------
    model : nn.Module
        The full CNN model.
    layer_name : str
        Name of the layer to extract feature maps from.
    """

    def __init__(self, model: nn.Module, layer_name: str):
        super().__init__()
        children = list(model.named_children())
        names, modules = zip(*children)

        if layer_name not in names:
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available layers: {list(names)}"
            )

        idx = names.index(layer_name)
        self.layers = nn.Sequential(*modules[: idx + 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _LogitsModel(nn.Module):
    """
    Takes feature maps from a specific layer and runs the remaining
    CNN layers to produce final class scores (logits).

    Used during Integrated Gradients to compute gradients of the class
    score with respect to intermediate feature maps.

    Parameters
    ----------
    model : nn.Module
        The full CNN model.
    layer_name : str
        Name of the layer the feature maps come from.
    """

    def __init__(self, model: nn.Module, layer_name: str):
        super().__init__()
        children = list(model.named_children())
        names, modules = zip(*children)

        if layer_name not in names:
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available layers: {list(names)}"
            )

        idx = names.index(layer_name)
        # Everything after the target layer except avgpool and fc,
        # which are handled separately to support models with different heads
        self.conv_layers = nn.Sequential(*modules[idx + 1: -2])
        self.avg_layer = modules[-2]
        self.fc_layer = modules[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.avg_layer(x)
        x = x.view(x.size(0), -1)  # flatten before the fully connected layer
        x = self.fc_layer(x)
        return x


class TorchModelWrapper:
    """
    Wraps any PyTorch CNN and provides a consistent interface for Visual-TCAV.

    Handles model loading, preprocessing, label management, and exposes
    methods to extract predictions, feature maps, and gradients at any layer.

    Labels and preprocessing are resolved in this order:
    1. Explicit ``labels`` / ``model_preprocess`` parameters (highest priority)
    2. torchvision ``_weights`` attribute on the model object
    3. Auto-lookup by ``model_name`` in the known torchvision weights map
    4. Safe defaults (Identity preprocessing, error for missing labels)

    Parameters
    ----------
    model_name : str
        A name for this model, used for caching and display.
        For known torchvision models (e.g. "resnet50"), labels and
        preprocessing are auto-loaded if not provided explicitly.
    model : nn.Module, optional
        A PyTorch model already loaded in memory.
    model_path : str, optional
        Path to a saved PyTorch model file (.pt or .pth).
    labels : list of str, optional
        List of class names indexed by model output position.
    labels_path : str, optional
        Path to a text file with one class name per line.
    model_preprocess : callable, optional
        Preprocessing function applied to images before inference.
    input_size : tuple, optional
        Expected input size as (C, H, W). Default is (3, 224, 224).
    batch_size : int, optional
        Default batch size for processing multiple images. Default: 32.

    Examples
    --------
    >>> import torchvision.models as models
    >>> resnet = models.resnet50(weights='DEFAULT')
    >>> wrapper = TorchModelWrapper(model_name="resnet50", model=resnet)
    >>> wrapper.info()
    """

    def __init__(
        self,
        model_name: str,
        model: nn.Module = None,
        model_path: str = None,
        labels: list = None,
        labels_path: str = None,
        model_preprocess=None,
        input_size: tuple = None,
        batch_size: int = 32,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size

        # --- Load model ---
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = torch.load(
                model_path, weights_only=False, map_location=self.device
            )
        else:
            raise ValueError(
                "Provide either 'model' (a PyTorch nn.Module) "
                "or 'model_path' (path to a saved model file)."
            )

        self.model.to(self.device)
        # eval() disables dropout and batchnorm updates during inference
        self.model.eval()

        # --- Resolve weights object for auto-loading ---
        # Try three sources in order of priority:
        # 1. model._weights attribute (set by torchvision when loading with weights=)
        # 2. known weights map keyed by model_name
        _weights_obj = None
        if (
            hasattr(self.model, "_weights")
            and self.model._weights is not None
            and hasattr(self.model._weights, "meta")
        ):
            _weights_obj = self.model._weights
        else:
            # Normalize model_name for lookup (e.g. "ResNet50" -> "resnet50")
            _name_normalized = model_name.lower().replace("-", "").replace("_", "")
            _weights_map = _get_torchvision_weights()
            # Try exact match first, then normalized
            if model_name in _weights_map:
                _weights_obj = _weights_map[model_name]
            elif _name_normalized in _weights_map:
                _weights_obj = _weights_map[_name_normalized]

        # --- Load preprocessing ---
        if model_preprocess is not None:
            self.model_preprocess = model_preprocess
        elif _weights_obj is not None and hasattr(_weights_obj, "transforms"):
            self.model_preprocess = _weights_obj.transforms()
        else:
            self.model_preprocess = nn.Identity()

        # --- Infer input size ---
        if input_size is not None:
            self.input_size = input_size
        elif _weights_obj is not None and hasattr(_weights_obj, "transforms"):
            dummy = torch.randn(3, 512, 512)
            processed = _weights_obj.transforms()(dummy)
            self.input_size = tuple(processed.shape)
        else:
            self.input_size = (3, 224, 224)

        # --- Load labels ---
        if labels is not None:
            self.labels = list(labels)
        elif labels_path is not None:
            with open(labels_path, "r") as f:
                self.labels = f.read().splitlines()
        elif _weights_obj is not None and "categories" in _weights_obj.meta:
            self.labels = list(_weights_obj.meta["categories"])
        else:
            raise ValueError(
                f"Could not auto-load labels for model '{model_name}'.\n"
                f"Provide either:\n"
                f"  labels=['class1', 'class2', ...]  (list of class names)\n"
                f"  labels_path='path/to/labels.txt'  (text file, one class per line)\n"
                f"For standard torchvision models, pass model_name as one of: "
                f"{list(_get_torchvision_weights().keys())}"
            )

        self.binary_classification = len(self.labels) == 2

    # -----------------------------------------------------------------------
    # Label utilities
    # -----------------------------------------------------------------------

    def id_to_label(self, idx: int) -> str:
        """
        Convert a class index to its human-readable name.

        Parameters
        ----------
        idx : int
            Class index (e.g. 340).

        Returns
        -------
        str
            Class name (e.g. "zebra").
        """
        return self.labels[idx]

    def label_to_id(self, label: str) -> int:
        """
        Convert a class name to its index.

        Parameters
        ----------
        label : str
            Class name (e.g. "zebra").

        Returns
        -------
        int
            Class index (e.g. 340).

        Raises
        ------
        ValueError
            If the label is not found in the model's class list.
        """
        if label not in self.labels:
            raise ValueError(
                f"Label '{label}' not found. "
                f"Check available classes with wrapper.info()."
            )
        return self.labels.index(label)

    # -----------------------------------------------------------------------
    # Model info
    # -----------------------------------------------------------------------

    def info(self) -> PrettyTable:
        """
        Print available layer names for use with Visual-TCAV.

        Deeper layers (closer to the output) capture higher-level concepts
        such as textures and object parts. For ResNet50, 'layer4' is a
        good starting point.

        Returns
        -------
        PrettyTable
            Formatted table with model name, number of classes, and layers.
        """
        layer_names = [
            name for name, module in self.model.named_children()
            if not isinstance(module, (nn.Linear, nn.Flatten))
            and name not in ("avgpool", "fc", "classifier")
        ]

        table = PrettyTable(
            title=f"Model: {self.model_name}",
            field_names=["N. classes", "Layers"],
        )
        for i, name in enumerate(layer_names):
            table.add_row([len(self.labels) if i == 0 else "", name])
        print(table)
        return table

    # -----------------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------------

    def get_predictions(self, data) -> torch.Tensor:
        """
        Run one or more images through the model and return softmax probabilities.

        Parameters
        ----------
        data : torch.Tensor or DataLoader
            A single image tensor of shape [1, C, H, W], or a DataLoader.

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape [N, num_classes].
        """
        softmax = nn.Softmax(dim=1)
        outputs = []

        if isinstance(data, DataLoader):
            with torch.no_grad():
                for batch_input, _ in data:
                    batch_input = batch_input.to(self.device)
                    outputs.append(softmax(self.model(batch_input)).detach().cpu())
            return torch.cat(outputs, dim=0)
        else:
            with torch.no_grad():
                data = data.to(self.device)
                return softmax(self.model(data)).detach().cpu()

    # -----------------------------------------------------------------------
    # Feature maps
    # -----------------------------------------------------------------------

    def get_feature_maps(self, imgs: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Extract feature maps at a specific layer.

        Parameters
        ----------
        imgs : torch.Tensor
            Input images of shape [batch, C, H, W].
        layer_name : str
            Name of the layer to extract from (e.g. 'layer4').

        Returns
        -------
        torch.Tensor
            Feature maps of shape [batch, channels, H, W].
        """
        imgs = imgs.to(self.device)
        f_model = _FeatureMapsModel(self.model, layer_name).to(self.device)
        with torch.no_grad():
            return f_model(imgs)

    def get_feature_maps_for_concept(
        self, concept_path: str, layer_name: str
    ) -> torch.Tensor:
        """
        Load all images from a concept folder and extract their feature maps.

        Parameters
        ----------
        concept_path : str
            Path to the concept folder (ImageFolder format).
        layer_name : str
            Name of the layer to extract from.

        Returns
        -------
        torch.Tensor
            Feature maps of shape [N, channels, H, W].
        """
        all_feature_maps = []
        for batch_imgs, _ in self._get_images_for_concept(concept_path):
            batch_imgs = batch_imgs.to(self.device)
            fmaps = self.get_feature_maps(batch_imgs, layer_name)
            all_feature_maps.append(fmaps.detach().cpu())
        return torch.cat(all_feature_maps, dim=0)

    # -----------------------------------------------------------------------
    # Logits
    # -----------------------------------------------------------------------

    def get_logits(
        self, feature_maps: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        """
        Run the second half of the CNN from a specific layer to the output.

        Parameters
        ----------
        feature_maps : torch.Tensor
            Feature maps from a specific layer. Shape: [batch, C, H, W].
        layer_name : str
            Name of the layer the feature maps came from.

        Returns
        -------
        torch.Tensor
            Logits (raw class scores) of shape [batch, num_classes].
        """
        if feature_maps.dim() == 3:
            feature_maps = feature_maps.unsqueeze(0)
        l_model = _LogitsModel(self.model, layer_name).to(self.device)
        return l_model(feature_maps)

    # -----------------------------------------------------------------------
    # Gradients
    # -----------------------------------------------------------------------

    def get_gradient_of_score(
        self,
        feature_maps: torch.Tensor,
        layer_name: str,
        target_class_index: int,
    ) -> torch.Tensor:
        """
        Compute the gradient of the target class score w.r.t. feature maps.

        Core computation for Integrated Gradients: measures how much each
        feature map value influences the target class prediction.
        Processed in batches to avoid memory issues.

        Parameters
        ----------
        feature_maps : torch.Tensor
            Interpolated feature maps. Shape: [steps, C, H, W].
        layer_name : str
            Name of the layer the feature maps came from.
        target_class_index : int
            Index of the class to explain.

        Returns
        -------
        torch.Tensor
            Gradients of shape [steps, C, H, W].
        """
        gradients = []
        batch_size = _safe_batch_size(len(feature_maps), desired=self.batch_size)

        for i in range(0, len(feature_maps), batch_size):
            batch = feature_maps[i: i + batch_size].to(self.device)
            inputs = batch.detach().clone().requires_grad_(True)
            logits = self.get_logits(inputs, layer_name)
            score = logits[:, target_class_index].sum()
            score.backward()
            gradients.append(inputs.grad.detach().cpu())

        return torch.cat(gradients, dim=0)

    # -----------------------------------------------------------------------
    # Image loading
    # -----------------------------------------------------------------------

    def _get_image_folder(
        self, concept_path: str, preprocess: bool = True
    ) -> datasets.ImageFolder:
        """
        Load images from a folder using torchvision ImageFolder.

        Expects the structure: concept_path/any_subfolder/image_files

        Parameters
        ----------
        concept_path : str
            Path to the concept folder.
        preprocess : bool
            Whether to apply model preprocessing. Default True.

        Returns
        -------
        datasets.ImageFolder
            Dataset ready to be wrapped in a DataLoader.
        """
        C, H, W = self.input_size

        if preprocess:
            transform = transforms.Compose([
                transforms.Resize(
                    (H, W),
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop((H, W)),
                transforms.ToTensor(),
                self.model_preprocess,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(
                    (H, W),
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop((H, W)),
                transforms.ToTensor(),
            ])

        return datasets.ImageFolder(root=concept_path, transform=transform)

    def _get_images_for_concept(self, concept_path: str) -> DataLoader:
        """
        Load concept images and return them as a DataLoader.

        Parameters
        ----------
        concept_path : str
            Path to the concept folder (ImageFolder format).

        Returns
        -------
        DataLoader
            Yields batches of preprocessed images.
        """
        image_folder = self._get_image_folder(concept_path)
        batch_size = _safe_batch_size(len(image_folder), desired=self.batch_size)
        # num_workers=0 required for Windows compatibility
        return DataLoader(
            image_folder, batch_size=batch_size, shuffle=False, num_workers=0
        )