"""
visual_tcav
-----------
Visual-TCAV: Concept-based Attribution and Saliency Maps for
Post-hoc Explainability in Image Classification.

Given an image and a human concept (e.g. "stripes"), Visual-TCAV produces:
- A concept map: a heatmap showing WHERE the CNN detected that concept
- An attribution score: measuring HOW MUCH the concept influenced the prediction

Quick start
-----------
>>> from visual_tcav import available_layers, LocalVisualTCAV
>>>
>>> # Step 1: inspect available layers before configuring
>>> available_layers("resnet50")
>>>
>>> # Step 2: instantiate with all configuration
>>> tcav = LocalVisualTCAV(
...     model="resnet50",
...     test_image_path="./zebra.jpg",
...     concept_names=["striped", "dotted"],
...     concept_base_dir="./concept_images",
...     random_dir="./concept_images/random",
...     layer_names=["layer4"],
...     cache_dir="./cache",
... )
>>> tcav.explain()
>>> tcav.plot()

Based on:
- De Santis et al., "Visual-TCAV: Concept-based Attribution and Saliency
    Maps for Post-hoc Explainability in Image Classification", TMLR 2026.
    https://openreview.net/forum?id=SLh00W5rhu

- Text-to-Concept extension by Daniele Di Santi (2025), based on:
    Moayeri et al., "Text-To-Concept (and Back) via Cross-Model Alignment",
    arXiv:2305.06386, 2023.
"""

# Main public API
from visual_tcav.local_tcav import LocalVisualTCAV
from visual_tcav.global_tcav import GlobalVisualTCAV

# Utility function — inspect model layers before instantiating
from visual_tcav.visual_tcav import available_layers

# Text-to-Concept extension
from visual_tcav.text_to_concept import TextToConcept
from visual_tcav.linear_aligner import LinearAligner

# Data classes for advanced users
from visual_tcav.utils import (
    Cav,
    ConceptLayer,
    Prediction,
    Predictions,
    Stat,
    CustomColormap,
    DEFAULT_COLORMAP,
)

# Advanced: explicit model wrapper for custom models
# Most users do not need this — pass model= to LocalVisualTCAV instead
from visual_tcav.model_wrapper import TorchModelWrapper

# Package metadata
__version__ = "1.0.0"
__license__ = "MIT"

__authors__ = [
    ("Antonio De Santis", "antonio.desantis@polimi.it"),
    ("Riccardo Campi", "riccardo.campi@polimi.it"),
    ("Matteo Bianchi", "matteo.bianchi@polimi.it"),
    ("Marco Brambilla", "marco.brambilla@polimi.it"),
    ("Sara Cavallini", "saracavallini01@gmail.com")
]

__all__ = [
    # Main classes
    "LocalVisualTCAV",
    "GlobalVisualTCAV",
    # Utility
    "available_layers",
    # Text-to-Concept extension
    "TextToConcept",
    "LinearAligner",
    # Data classes
    "Cav",
    "ConceptLayer",
    "Prediction",
    "Predictions",
    "Stat",
    "CustomColormap",
    "DEFAULT_COLORMAP",
    # Advanced
    "TorchModelWrapper",
]
