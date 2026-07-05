"""
visual_tcav
-----------
Visual-TCAV: Concept-based Attribution and Saliency Maps for
Explainable AI in PyTorch.

Given an image and a human concept (e.g. "stripes"), Visual-TCAV produces:
- A concept map: a heatmap showing WHERE the CNN detected that concept
- An attribution score: measuring HOW MUCH the concept influenced the prediction

This package implements Visual-TCAV in PyTorch, based on:
    De Santis et al., "Visual-TCAV: Concept-based Attribution and Saliency
    Maps for Post-hoc Explainability in Image Classification", 2025.
    https://github.com/DataSciencePolimi/Visual-TCAV

It also includes the Text-to-Concept extension by Daniele Di Santi (2025),
which enables CAV generation from plain text instead of concept images,
using CLIP and a trained Linear Aligner.

Basic usage
-----------
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
...     layer_names=["layer4"],
... )
>>> tcav.predict().info()
>>> tcav.explain()
>>> tcav.plot()

Text-to-Concept usage
---------------------
>>> from visual_tcav import TextToConcept
>>>
>>> t2c = TextToConcept(model_wrapper=wrapper)
>>> t2c.load_aligner("./aligners/resnet50_layer4.pt")
>>> cav = t2c.get_cav_from_text("stripes", layer_name="layer4")
"""

from visual_tcav.model_wrapper import TorchModelWrapper
from visual_tcav.local_tcav import LocalVisualTCAV
from visual_tcav.global_tcav import GlobalVisualTCAV
from visual_tcav.text_to_concept import TextToConcept
from visual_tcav.linear_aligner import LinearAligner
from visual_tcav.utils import (
    Cav,
    ConceptLayer,
    Prediction,
    Predictions,
    Stat,
    CustomColormap,
    DEFAULT_COLORMAP,
)

# Package metadata
__version__ = "0.1.0"
__author__ = "Sara Cavallini"
__email__ = ""
__license__ = "MIT"

# Defines the public API — what `from visual_tcav import *` exposes
__all__ = [
    # Main classes — what most users will import
    "LocalVisualTCAV",
    "GlobalVisualTCAV",
    "TorchModelWrapper",
    # Text-to-Concept extension
    "TextToConcept",
    "LinearAligner",
    # Data classes — for advanced users
    "Cav",
    "ConceptLayer",
    "Prediction",
    "Predictions",
    "Stat",
    "CustomColormap",
    "DEFAULT_COLORMAP",
]