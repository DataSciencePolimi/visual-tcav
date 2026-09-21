# visual-tcav

[![PyPI version](https://badge.fury.io/py/visual-tcav.svg)](https://badge.fury.io/py/visual-tcav)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/visual-tcav/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch package for concept-based attribution and saliency maps in Explainable AI.

Given a test image and a human-understandable concept (e.g. *"stripes"*), **visual-tcav** produces:
- A **concept map** — a heatmap showing *where* a CNN detected that concept in the image
- An **attribution score** — measuring *how much* the concept influenced the model's prediction

Based on:
<!-- > De Santis et al., *Visual-TCAV: Concept-based Attribution and Saliency Maps for Post-hoc Explainability in Image Classification*, 2025. [arXiv:2411.05698](https://arxiv.org/abs/2411.05698) -->
> De Santis et al., *Visual-TCAV: Concept-based Attribution and Saliency Maps for Post-hoc Explainability in Image Classification*, **Transactions on Machine Learning Research (TMLR), 2025**. [OpenReview](https://openreview.net/forum?id=SLh00W5rhu)

---

## Installation

**pip** (recommended):
```bash
pip install visual-tcav
```

**conda**:
```bash
conda install -c conda-forge visual-tcav
```

**From source**:
```bash
git clone https://github.com/saracavallini01/visual-tcav.git
cd visual-tcav
pip install -e .
```

**Without installation**:
```bash
git clone https://github.com/saracavallini01/visual-tcav.git
cd visual-tcav
pip install -r requirements.txt
```

**Text-to-Concept extension** (requires CLIP):
```bash
pip install visual-tcav[text-to-concept]
pip install git+https://github.com/openai/CLIP.git
```

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- torchvision >= 0.15.0

---

## Quick start

```python
from visual_tcav import available_layers, LocalVisualTCAV

# Step 1 — inspect available layers before configuring
available_layers("resnet50")

# Step 2 — instantiate with full configuration
tcav = LocalVisualTCAV(
    model="resnet50",
    test_image_path="./examples/data/test_images/zebra.jpg",
    concept_names=["striped", "dotted"],
    concept_base_dir="./examples/data/concept_images",
    random_dir="./examples/data/concept_images/random",
    layer_names=["layer4"],
    cache_dir="./.cache",
)

# Step 3 — explain and visualize
tcav.explain()
tcav.plot()
```

---

## Concept image folder structure

```
data/
├── concept_images/
│   ├── striped/       ← ~50 images per concept (e.g. from DTD dataset)
│   ├── dotted/
│   └── random/        ← ~50 random reference images
└── test_images/
    └── zebra.jpg
```

All paths are individually configurable — no folder structure is imposed.

---

## Key features

| Feature | Usage |
|---|---|
| Inspect layers before instantiating | `available_layers("resnet50")` |
| Load model from string | `LocalVisualTCAV(model="resnet50", ...)` |
| Load model from nn.Module | `LocalVisualTCAV(model=my_model, model_name="resnet50", ...)` |
| Skip explicit predict() | `explain()` calls it automatically |
| Use cache (default) | `tcav.explain()` |
| Force full recompute | `tcav.explain(force_recompute=True)` |
| Clear cache | `tcav.clear_cache()` |
| Disable cache entirely | `LocalVisualTCAV(..., cache_dir=None)` |
| Analyze multiple layers | `layer_names=["layer2", "layer3", "layer4"]` |
| Custom CAV function | `LocalVisualTCAV(cav_fn=my_function, ...)` |
| Global explanation | `GlobalVisualTCAV(test_images_dir=..., ...)` |
| Save figures | `plot(save_path="./output.png")` |
| Command-line interface | `visual-tcav local --model resnet50 --image ...` |

---

## GlobalVisualTCAV

Run the explanation pipeline on a folder of images and get statistical
summaries (mean, standard deviation, 95% confidence interval):

```python
from visual_tcav import GlobalVisualTCAV

tcav = GlobalVisualTCAV(
    model="resnet50",
    test_images_dir="./data/test_images/zebra",
    concept_names=["striped", "dotted"],
    concept_base_dir="./data/concept_images",
    random_dir="./data/concept_images/random",
    layer_names=["layer4"],
    max_test_images=50,
)

tcav.explain()
tcav.statsInfo()
tcav.plot()
```

---

## Custom CAV function

Replace the default centroid difference method with any custom function:

```python
from visual_tcav import LocalVisualTCAV, Cav
import torch

def normalized_cav(concept_features, random_features):
    direction = (
        torch.mean(concept_features, dim=0)
        - torch.mean(random_features, dim=0)
    )
    return Cav(direction=direction / (torch.norm(direction) + 1e-10))

tcav = LocalVisualTCAV(
    model="resnet50",
    cav_fn=normalized_cav,
    ...
)
```

---

## Command-line interface

```bash
# Inspect available layers
visual-tcav info --model resnet50

# Local explanation
visual-tcav local \
    --model resnet50 \
    --image ./data/zebra.jpg \
    --concepts striped dotted \
    --concept-dir ./data/concept_images \
    --layers layer4 \
    --output ./results

# Global explanation
visual-tcav global \
    --model resnet50 \
    --images-dir ./data/test_images/zebra \
    --concepts striped dotted \
    --concept-dir ./data/concept_images \
    --layers layer4 \
    --output ./results
```

---

## Text-to-Concept extension

Generate CAVs from plain text descriptions instead of concept images,
using CLIP and a trained Linear Aligner:

```python
from visual_tcav import TextToConcept, LocalVisualTCAV

tcav = LocalVisualTCAV(model="resnet50", ...)
t2c = TextToConcept(model_wrapper=tcav.model_wrapper)
t2c.load_aligner("./aligners/resnet50_layer4.pt")

cav = t2c.get_cav_from_text("stripes", layer_name="layer4")
```

Based on:
> Di Santi, D., *Text-to-Concept Extension for Visual-TCAV*, Master's Thesis, Politecnico di Milano, 2025.
> Moayeri et al., *Text-To-Concept (and Back) via Cross-Model Alignment*, ICML 2023.

---

## Demo notebook

See [`examples/resnet50_demo.ipynb`](examples/resnet50_demo.ipynb) for a
complete end-to-end demonstration with ResNet50 and DTD concept images,
covering all features with increasing complexity.

---

## Citation

If you use this package in your research, please cite:

```bibtex
@article{
    santis2026visualtcav,
    title={Visual-{TCAV}: Concept-based Attribution and Saliency Maps for Post-hoc Explainability in Image Classification},
    author={Antonio De Santis and Riccardo Campi and Matteo Bianchi and Marco Brambilla},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2026},
    url={https://openreview.net/forum?id=SLh00W5rhu},
    note={}
}
```

---

## License

This project is licensed under the MIT License — see the
[LICENSE](LICENSE) file for details.
