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
from PIL import Image
from prettytable import PrettyTable

sys.dont_write_bytecode = True


# ---------------------------------------------------------------------------
# Helper: compute a safe batch size
# ---------------------------------------------------------------------------

def _safe_batch_size(n_samples: int, desired: int = 32) -> int:
    """
    Find the largest batch size <= desired that evenly divides n_samples.

    This avoids issues with uneven batches during gradient computation.

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


# ---------------------------------------------------------------------------
# Helper models: extract feature maps or logits at a specific layer
# ---------------------------------------------------------------------------

class _FeatureMapsModel(nn.Module):
    """
    A helper model that runs the input through a CNN up to a specific layer
    and returns the feature maps (activations) at that layer.

    Think of it as cutting the CNN at a certain point and reading
    what comes out there.

    Parameters
    ----------
    model : nn.Module
        The full CNN model (e.g. ResNet50).
    layer_name : str
        Name of the layer where we want to extract feature maps.
        Must be a top-level child of the model (e.g. 'layer4' for ResNet50).
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
        # Keep only layers up to and including the target layer
        self.layers = nn.Sequential(*modules[: idx + 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _LogitsModel(nn.Module):
    """
    A helper model that takes feature maps from a specific layer as input
    and runs them through the REST of the CNN to produce final class scores
    (logits).

    This is the opposite of _FeatureMapsModel: instead of cutting the CNN
    at a layer and reading what comes out, we START from that layer and
    run everything after it.

    This is needed for Integrated Gradients: we need to compute gradients
    of the final class score with respect to the feature maps.

    Parameters
    ----------
    model : nn.Module
        The full CNN model (e.g. ResNet50).
    layer_name : str
        Name of the layer where the feature maps come from.
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

        # Everything AFTER the target layer, except the last two
        # (avgpool and fc in ResNet50)
        self.conv_layers = nn.Sequential(*modules[idx + 1: -2])

        # The final average pooling and fully connected layer
        self.avg_layer = modules[-2]
        self.fc_layer = modules[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.avg_layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layer(x)
        return x


# ---------------------------------------------------------------------------
# TorchModelWrapper
# ---------------------------------------------------------------------------

class TorchModelWrapper:
    """
    Wraps any PyTorch CNN model and provides a consistent interface
    for Visual-TCAV to use.

    Instead of Visual-TCAV knowing the details of every possible model,
    it just talks to this wrapper. The wrapper handles all the
    model-specific details internally.

    Parameters
    ----------
    model_name : str
        A name for this model, used for caching and display.
        Example: "resnet50"
    model : nn.Module, optional
        A PyTorch model object already loaded in memory.
        Either model or model_path must be provided.
    model_path : str, optional
        Path to a saved PyTorch model file (.pt or .pth).
        Either model or model_path must be provided.
    labels : list of str, optional
        List of class names. Index must match model output.
        Example: ["tench", "goldfish", ..., "zebra", ...]
    labels_path : str, optional
        Path to a text file with one class name per line.
    model_preprocess : callable, optional
        Preprocessing function to apply to images before passing
        them to the model (e.g. normalization).
        If not provided, will try to read it from model._weights.
    input_size : tuple, optional
        Expected input size as (C, H, W). Example: (3, 224, 224).
        If not provided, will be inferred automatically.
    batch_size : int, optional
        Default batch size for processing multiple images. Default: 32.

    Examples
    --------
    >>> import torchvision.models as models
    >>> from visual_tcav.model_wrapper import TorchModelWrapper
    >>>
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
        # Device: use GPU if available, otherwise CPU
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
                "You must provide either 'model' (a PyTorch model object) "
                "or 'model_path' (path to a saved model file)."
            )
        self.model.to(self.device)
        self.model.eval()  # set to evaluation mode (disables dropout, batchnorm updates)

        # --- Load preprocessing function ---
        if model_preprocess is not None:
            self.model_preprocess = model_preprocess
        elif (
            hasattr(self.model, "_weights")
            and self.model._weights is not None
            and hasattr(self.model._weights, "transforms")
        ):
            # torchvision models store their preprocessing in _weights
            self.model_preprocess = self.model._weights.transforms()
        else:
            # No preprocessing: just pass the tensor as-is
            self.model_preprocess = nn.Identity()

        # --- Infer input size ---
        if input_size is not None:
            self.input_size = input_size
        elif (
            hasattr(self.model, "_weights")
            and self.model._weights is not None
            and hasattr(self.model._weights, "transforms")
        ):
            # Run a dummy image through preprocessing to get the size
            dummy = torch.randn(3, 512, 512)
            processed = self.model._weights.transforms()(dummy)
            self.input_size = tuple(processed.shape)  # (C, H, W)
        else:
            # Default to standard ImageNet size
            self.input_size = (3, 224, 224)

        # --- Load labels ---
        if labels is not None:
            self.labels = labels
        elif labels_path is not None:
            with open(labels_path, "r") as f:
                self.labels = f.read().splitlines()
        elif (
            hasattr(self.model, "_weights")
            and self.model._weights is not None
            and "categories" in self.model._weights.meta
        ):
            # torchvision models store their class names in _weights.meta
            self.labels = self.model._weights.meta["categories"]
        else:
            raise ValueError(
                "You must provide either 'labels' (a list of class names) "
                "or 'labels_path' (path to a text file with class names)."
            )

        # Binary classification flag
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
                f"Check the model's class list with wrapper.info()."
            )
        return self.labels.index(label)

    # -----------------------------------------------------------------------
    # Info
    # -----------------------------------------------------------------------

    def info(self) -> PrettyTable:
        """
        Print a table showing the model name, number of classes,
        and all available layer names.

        This is useful for deciding which layers to analyze with Visual-TCAV.
        Deeper layers (closer to the output) tend to represent higher-level
        concepts like "stripes" or "wheels".

        Returns
        -------
        PrettyTable
            A formatted table with model info and layer names.

        Examples
        --------
        >>> wrapper.info()
        +----------------------------+
        |       Model: resnet50      |
        +------------+---------------+
        | N. classes |     Layers    |
        +------------+---------------+
        |    1000    |    layer1     |
        |            |    layer2     |
        |            |    layer3     |
        |            |    layer4     |
        +------------+---------------+
        """
        # Get only convolutional/feature layers (exclude avgpool and fc)
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
            table.add_row([
                len(self.labels) if i == 0 else "",
                name,
            ])
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
            A single image tensor of shape [1, C, H, W], or a DataLoader
            that yields batches of images.

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
                    batch_output = softmax(self.model(batch_input))
                    outputs.append(batch_output.detach().cpu())
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
        Extract feature maps (internal activations) at a specific layer.

        Feature maps are what the CNN "sees" at a given layer.
        For example, at 'layer4' of ResNet50, the feature maps have
        shape [batch, 2048, 7, 7] — 2048 channels, 7x7 spatial grid.

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
            feature_maps = f_model(imgs)

        return feature_maps

    def get_feature_maps_for_concept(
        self, concept_path: str, layer_name: str
    ) -> torch.Tensor:
        """
        Load all images from a concept folder and extract their feature maps.

        This is the first step in computing a CAV: you load all your
        concept images (e.g. 50 striped texture photos) and get their
        internal CNN representations.

        Parameters
        ----------
        concept_path : str
            Path to the folder containing concept images.
            The folder must contain a subfolder with the images
            (ImageFolder format).
        layer_name : str
            Name of the layer to extract from.

        Returns
        -------
        torch.Tensor
            Feature maps of shape [N, channels, H, W] where N is the
            number of concept images.
        """
        images = self._get_images_for_concept(concept_path)
        feature_maps = self.get_feature_maps(
            next(iter(images))[0], layer_name
        )

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
        Given feature maps from a specific layer, run the rest of the
        CNN and return the final class scores (logits).

        This is the second half of the CNN, used during Integrated Gradients
        computation.

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
        Compute the gradient of the target class score with respect
        to the feature maps.

        This is the core computation behind Integrated Gradients.
        The gradient tells us: "if I change this feature map slightly,
        how much does the class score change?"

        Processed in batches to avoid running out of memory.

        Parameters
        ----------
        feature_maps : torch.Tensor
            Feature maps of shape [steps, C, H, W] — the interpolated
            feature maps generated during Integrated Gradients computation.
        layer_name : str
            Name of the layer the feature maps came from.
        target_class_index : int
            Index of the class we want to explain (e.g. 340 for zebra).

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

            # Forward pass: get logits for this batch
            logits = self.get_logits(inputs, layer_name)

            # Select score for target class only
            score = logits[:, target_class_index].sum()

            # Backward pass: compute gradients
            score.backward()

            gradients.append(inputs.grad.detach().cpu())

        return torch.cat(gradients, dim=0)

    # -----------------------------------------------------------------------
    # Image loading utilities
    # -----------------------------------------------------------------------

    def _get_image_folder(
        self, concept_path: str, preprocess: bool = True
    ) -> datasets.ImageFolder:
        """
        Load images from a folder using torchvision ImageFolder.

        ImageFolder expects this structure:
            concept_path/
            └── any_subfolder_name/
                ├── image1.jpg
                ├── image2.jpg
                └── ...

        Parameters
        ----------
        concept_path : str
            Path to the concept folder.
        preprocess : bool
            Whether to apply model preprocessing. Default True.

        Returns
        -------
        datasets.ImageFolder
            A PyTorch dataset ready to be wrapped in a DataLoader.
        """
        C, H, W = self.input_size

        if preprocess:
            transform = transforms.Compose([
                transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((H, W)),
                transforms.ToTensor(),
                self.model_preprocess,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
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
            A DataLoader that yields batches of preprocessed images.
        """
        image_folder = self._get_image_folder(concept_path)
        batch_size = _safe_batch_size(len(image_folder), desired=self.batch_size)
        return DataLoader(image_folder, batch_size=batch_size, shuffle=False)