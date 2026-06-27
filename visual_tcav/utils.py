"""
utils.py
--------
Helper classes and functions used across the visual_tcav package.
"""

import os
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt, cm as cm
from matplotlib.colors import LinearSegmentedColormap
from prettytable import PrettyTable

sys.dont_write_bytecode = True


# ---------------------------------------------------------------------------
# Small math utilities
# ---------------------------------------------------------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Cosine similarity measures how similar two vectors are, regardless
    of their length. Returns 1.0 if identical direction, 0.0 if
    perpendicular, -1.0 if opposite.

    Parameters
    ----------
    vec1 : np.ndarray
        First vector.
    vec2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine similarity value between -1 and 1.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def nth_highest_index(arr: list, n: int) -> int:
    """
    Return the index of the n-th highest value in an array.

    Parameters
    ----------
    arr : list
        Input array of values.
    n : int
        Rank to find (1 = highest, 2 = second highest, etc.).

    Returns
    -------
    int
        Index of the n-th highest value.
    """
    indexed_arr = list(enumerate(arr))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    return sorted_arr[n - 1][0]


def contraharmonic_mean(arr: torch.Tensor, axis: tuple = (0, 1)) -> torch.Tensor:
    """
    Compute the contraharmonic mean of a tensor along given axes.

    The contraharmonic mean is defined as sum(x^2) / sum(x).
    It is used in Visual-TCAV to compute the concept emblem (scale factor)
    from the concept activations.

    A small epsilon (1e-10) is added to the denominator to avoid
    division by zero.

    Parameters
    ----------
    arr : torch.Tensor
        Input tensor.
    axis : tuple
        Axes along which to compute the mean.

    Returns
    -------
    torch.Tensor
        Contraharmonic mean values.
    """
    x = torch.square(arr)
    numerator = torch.sum(x, dim=axis)
    denominator = torch.sum(arr, dim=axis)
    return torch.divide(numerator, denominator + 1e-10)


# ---------------------------------------------------------------------------
# Cav
# ---------------------------------------------------------------------------

class Cav:
    """
    Stores all data related to a Concept Activation Vector (CAV).

    A CAV represents a concept (e.g. 'stripes') as a direction in the
    internal space of a CNN layer. It is computed as the difference
    between the mean activation of concept images and the mean activation
    of random (negative) images at a given layer.

    Attributes
    ----------
    concept_centroid : torch.Tensor or None
        Mean activation vector of concept images (positive centroid).
    negative_centroid : torch.Tensor or None
        Mean activation vector of random images (negative centroid).
    direction : torch.Tensor or None
        CAV direction = concept_centroid - negative_centroid.
    concept_emblem : torch.Tensor or None
        Scale factor used to normalize the concept map.
        Computed from concept image activations using contraharmonic mean.
    """

    def __init__(
        self,
        concept_centroid: torch.Tensor = None,
        negative_centroid: torch.Tensor = None,
        direction: torch.Tensor = None,
        concept_emblem: torch.Tensor = None,
    ):
        self.concept_centroid = concept_centroid
        self.negative_centroid = negative_centroid
        self.direction = direction
        self.concept_emblem = concept_emblem

    def cpu(self) -> "Cav":
        """Move all tensors to CPU."""
        if self.concept_centroid is not None:
            self.concept_centroid = self.concept_centroid.detach().cpu()
        if self.negative_centroid is not None:
            self.negative_centroid = self.negative_centroid.detach().cpu()
        if self.direction is not None:
            self.direction = self.direction.detach().cpu()
        if self.concept_emblem is not None:
            self.concept_emblem = self.concept_emblem.detach().cpu()
        return self

    def to(self, device: torch.device) -> "Cav":
        """Move all tensors to the specified device (CPU or GPU)."""
        if self.concept_centroid is not None:
            self.concept_centroid = self.concept_centroid.detach().to(device)
        if self.negative_centroid is not None:
            self.negative_centroid = self.negative_centroid.detach().to(device)
        if self.direction is not None:
            self.direction = self.direction.detach().to(device)
        if self.concept_emblem is not None:
            self.concept_emblem = self.concept_emblem.detach().to(device)
        return self

    def __str__(self) -> str:
        return f"Cav | concept_emblem: {self.concept_emblem}"

    def __repr__(self) -> str:
        return f"Cav | concept_emblem: {self.concept_emblem}"


# ---------------------------------------------------------------------------
# ConceptLayer
# ---------------------------------------------------------------------------

class ConceptLayer:
    """
    Stores all computed data for a single (concept, layer) pair.

    When Visual-TCAV analyzes concept 'stripes' at layer 'layer4',
    all results are stored in one ConceptLayer object.

    Attributes
    ----------
    cav : Cav
        The Concept Activation Vector for this (concept, layer) pair.
    concept_map : torch.Tensor or None
        The spatial heatmap showing where the concept appears in the
        test image. Shape: [H, W].
    attributions : dict
        Attribution scores for each target class.
        Keys are class indices (int), values are scalar tensors.
        Example: {0: tensor(0.22), 1: tensor(0.05)}
    """

    def __init__(self, cav: Cav = None):
        self.cav = cav if cav is not None else Cav()
        self.concept_map = None
        self.attributions = {}

    def __str__(self) -> str:
        return self.cav.__str__()

    def __repr__(self) -> str:
        return self.cav.__repr__()


# ---------------------------------------------------------------------------
# Prediction / Predictions
# ---------------------------------------------------------------------------

class Prediction:
    """
    Stores the result for a single predicted class.

    Attributes
    ----------
    class_name : str
        Human-readable class name (e.g. 'zebra').
    class_index : int
        Class index in the model's output (e.g. 340).
    confidence : float
        Softmax probability for this class (between 0 and 1).
    """

    def __init__(
        self,
        class_name: str = None,
        class_index: int = None,
        confidence: float = None,
    ):
        self.class_name = class_name
        self.class_index = class_index
        self.confidence = confidence

    def __str__(self) -> str:
        return (
            f"Class: {self.class_name} "
            f"(index={self.class_index}, "
            f"confidence={self.confidence:.4f})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class Predictions:
    """
    Stores all predictions for a test image and provides display utilities.

    Attributes
    ----------
    predictions : list of list of Prediction
        Outer list = images (usually just 1), inner list = top-k classes.
    test_image_path : str
        Full path to the test image file.
    test_image_filename : str
        Just the filename (e.g. 'zebra.jpg').
    model_name : str
        Name of the model used for prediction.
    """

    def __init__(
        self,
        predictions: list,
        test_image_path: str,
        model_name: str,
    ):
        self.predictions = predictions
        self.test_image_path = test_image_path
        self.test_image_filename = os.path.basename(test_image_path)
        self.model_name = model_name

    def info(self, num_of_classes: int = 3) -> PrettyTable:
        """
        Print a formatted table with the top predicted classes.

        Parameters
        ----------
        num_of_classes : int
            How many top classes to show. Default is 3.

        Returns
        -------
        PrettyTable
            A formatted table with image name, class name, and confidence.
        """
        table = PrettyTable(
            title=f"Model: {self.model_name}",
            field_names=["Image", "Class name", "Confidence"],
            float_format=".4",
        )
        for i in range(min(num_of_classes, len(self.predictions[0]))):
            table.add_row([
                self.test_image_filename if i == 0 else "",
                self.predictions[0][i].class_name,
                f"{self.predictions[0][i].confidence:.4f}",
            ])
        print(table)
        return table

    def __getitem__(self, item):
        return self.predictions[item]

    def __str__(self) -> str:
        return f"Predictions for {self.test_image_filename} using {self.model_name}"

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Stat
# ---------------------------------------------------------------------------

class Stat:
    """
    Stores attribution statistics for the Global Explainer.

    The Global Explainer computes attribution scores across many images
    (e.g. 50 photos of zebras) and summarizes them with mean, standard
    deviation, and a 95% confidence interval.

    Attributes
    ----------
    attributions : list
        List of attribution scores (one per test image).
    mean : torch.Tensor
        Mean attribution score across all images.
    std : torch.Tensor
        Standard deviation of attribution scores.
    n : int
        Number of images.
    std_err : torch.Tensor
        Standard error = std / sqrt(n).
    begin : torch.Tensor
        Lower bound of the 95.45% confidence interval (mean - 2*std_err).
        Clipped to 0 with ReLU (attributions cannot be negative).
    end : torch.Tensor
        Upper bound of the 95.45% confidence interval (mean + 2*std_err).
    """

    def __init__(self, attributions: list):
        self.attributions = attributions
        self.mean = torch.mean(torch.tensor(attributions, dtype=torch.float32))
        self.std = torch.tensor(np.std(attributions, ddof=1), dtype=torch.float32)
        self.n = len(attributions)
        self.std_err = self.std / torch.sqrt(torch.tensor(self.n, dtype=torch.float32))
        self.begin = torch.relu(self.mean - self.std_err * 2)
        self.end = self.mean + self.std_err * 2


# ---------------------------------------------------------------------------
# CustomColormap
# ---------------------------------------------------------------------------

class CustomColormap:
    """
    A custom colormap for visualizing concept maps as heatmaps.

    The default colormap makes low-activation areas black (transparent)
    and high-activation areas use the jet colormap (blue → red).
    This makes it easy to overlay the heatmap on top of the original image.

    Attributes
    ----------
    nodes : list of float
        Positions along the colormap (between 0 and 1).
    colors : list of tuple
        RGBA colors at each node position.
    min : float
        Minimum value for colormap scaling.
    max : float
        Maximum value for colormap scaling.
    alpha : float
        Transparency of the heatmap overlay (0=transparent, 1=opaque).
    """

    def __init__(
        self,
        nodes: list = None,
        colors: list = None,
        min: float = 0.0,
        max: float = 1.0,
        alpha: float = 0.6,
    ):
        if nodes is None or colors is None:
            raise ValueError("Both nodes and colors must be provided.")
        if len(nodes) != len(colors):
            raise ValueError(
                f"nodes and colors must have the same length, "
                f"got {len(nodes)} and {len(colors)}."
            )
        if min >= max:
            raise ValueError(
                f"min ({min}) must be strictly less than max ({max})."
            )
        self.nodes = nodes
        self.colors = colors
        self.min = min
        self.max = max
        self.alpha = alpha

    def get_linear_segmented_colormap(self) -> LinearSegmentedColormap:
        """Build and return a matplotlib LinearSegmentedColormap."""
        return LinearSegmentedColormap.from_list(
            "custom", list(zip(self.nodes, self.colors))
        )

    def imshow(self, heatmap: np.ndarray) -> None:
        """
        Display a heatmap using this colormap.

        Parameters
        ----------
        heatmap : np.ndarray
            2D array of activation values to display.
        """
        plt.imshow(
            heatmap,
            cmap=self.get_linear_segmented_colormap(),
            alpha=self.alpha,
            vmin=self.min,
            vmax=self.max,
        )
        plt.clim(self.min, self.max)

    def get_min(self) -> float:
        """Return the minimum colormap value."""
        return self.min

    def get_max(self) -> float:
        """Return the maximum colormap value."""
        return self.max

    def get_alpha(self) -> float:
        """Return the colormap alpha (transparency)."""
        return self.alpha


# ---------------------------------------------------------------------------
# Default colormap used across the package
# ---------------------------------------------------------------------------

_original_colormap = cm.jet

DEFAULT_COLORMAP = CustomColormap(
    nodes=[0.0, 0.05] + [i for i in np.linspace(0.1, 1.0, 100)],
    colors=(
        [(0, 0, 0, 1), (0, 0, 0, 1)]
        + [_original_colormap(i) for i in np.linspace(0.15, 1.0, 100)]
    ),
    min=0.0,
    max=1.0,
    alpha=0.6,
)