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
# Math utilities
# ---------------------------------------------------------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

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

    Defined as sum(x^2) / sum(x). Used to compute the concept emblem
    (scale factor for concept map normalization) from concept activations.
    Epsilon avoids division by zero.

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
    numerator = torch.sum(torch.square(arr), dim=axis)
    denominator = torch.sum(arr, dim=axis)
    return torch.divide(numerator, denominator + 1e-10)


# ---------------------------------------------------------------------------
# Cav
# ---------------------------------------------------------------------------

class Cav:
    """
    Stores all data related to a Concept Activation Vector (CAV).

    Attributes
    ----------
    concept_centroid : torch.Tensor or None
        Mean activation vector of concept images.
    negative_centroid : torch.Tensor or None
        Mean activation vector of random images.
    direction : torch.Tensor or None
        CAV direction = concept_centroid - random_centroid.
    concept_emblem : torch.Tensor or None
        Scale factor used to normalize concept maps.
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
        """Move all tensors to the specified device."""
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

    Attributes
    ----------
    cav : Cav
        The Concept Activation Vector for this pair.
    concept_map : torch.Tensor or None
        Spatial heatmap showing where the concept appears. Shape: [H, W].
    attributions : dict
        Attribution scores per class. Keys: class indices, values: scalars.
    """

    def __init__(self, cav: "Cav" = None):
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
        Class index in the model output (e.g. 340).
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
    Stores all predictions for a test image.

    Attributes
    ----------
    predictions : list of list of Prediction
        Outer list = images, inner list = top-k classes.
    test_image_path : str
        Full path to the test image file.
    test_image_filename : str
        Filename only (e.g. 'zebra.jpg').
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
            Formatted table with image name, class name, and confidence.
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

    Computes mean, standard deviation, and a 95.45% confidence interval
    (mean +/- 2 * standard error) across a list of attribution scores.
    The lower bound is clipped to 0 since attributions are non-negative.

    Uses torch primitives throughout to avoid unnecessary numpy conversions,
    which would force data transfers between GPU and CPU on GPU machines.

    Attributes
    ----------
    attributions : list
        Raw attribution scores (one per test image).
    mean : torch.Tensor
        Mean attribution score.
    std : torch.Tensor
        Standard deviation (ddof=1 for unbiased estimate).
    n : int
        Number of images.
    std_err : torch.Tensor
        Standard error = std / sqrt(n).
    begin : torch.Tensor
        Lower bound of the confidence interval, clipped to 0.
    end : torch.Tensor
        Upper bound of the confidence interval.
    """

    def __init__(self, attributions: list):
        self.attributions = attributions
        t = torch.tensor(attributions, dtype=torch.float32)
        self.mean = torch.mean(t)
        # ddof=1 for unbiased standard deviation estimate
        self.std = torch.std(t, correction=1)
        self.n = len(attributions)
        self.std_err = self.std / torch.sqrt(torch.tensor(self.n, dtype=torch.float32))
        # ReLU clips the lower bound to 0 — attributions cannot be negative
        self.begin = torch.relu(self.mean - self.std_err * 2)
        self.end = self.mean + self.std_err * 2


# ---------------------------------------------------------------------------
# CustomColormap
# ---------------------------------------------------------------------------

class CustomColormap:
    """
    Custom colormap for visualizing concept maps as heatmaps.

    Parameters
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
        Transparency of the overlay (0=transparent, 1=opaque).
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
            raise ValueError(f"min ({min}) must be strictly less than max ({max}).")

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
        return self.min

    def get_max(self) -> float:
        return self.max

    def get_alpha(self) -> float:
        return self.alpha


# ---------------------------------------------------------------------------
# Default colormap instance
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