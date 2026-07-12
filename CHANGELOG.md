# Changelog

All notable changes to visual-tcav will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-12

### Added
- `LocalVisualTCAV`: explains a single image using Visual-TCAV
- `GlobalVisualTCAV`: explains a class of images with attribution statistics
- `TorchModelWrapper`: wraps any PyTorch CNN for use with Visual-TCAV
- `TextToConcept`: generates CAVs from plain text using CLIP (Daniele Di Santi)
- `LinearAligner`: maps CLIP embeddings to CNN feature space (Daniele Di Santi)
- CAV computation with Global Average Pooling
- Concept map generation (GradCAM-style weighted feature maps)
- Attribution scores via Integrated Gradients
- Caching of CAVs and random activations with joblib
- `model_wrapper.info()`: displays available CNN layers
- Support for ResNet50 pretrained on ImageNet
- Full NumPy-style docstrings on all classes and functions
- Type hints throughout