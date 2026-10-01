# Changelog

All notable changes to visual-tcav will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-09-30

### Added
- `GlobalVisualTCAV(target_class=...)` and `visual-tcav global --target-class`
  to compute attributions for one fixed class across all test images.
  Without it, scores are aggregated by prediction rank as before.

### Changed
- Concept maps are now normalized with the concept emblem of the reference
  implementation (median positive and negative emblem). Maps are scaled to
  [0, 1] relative to the concept and random images instead of a fixed factor.
- Integrated Gradients use `m_steps + 1` interpolation points and the
  trapezoidal rule, matching the reference implementation.
- Attribution scores are computed as in the reference implementation:
  rectified, max-normalized CAV direction and rescaling to the normalized
  logit gap. Scores are now non-negative and directly comparable across
  classes and images. Values differ from previous versions.
- Cached CAVs from previous versions are recomputed automatically.
- The concept emblem is computed in chunks and the random feature maps are
  released after each layer, reducing peak memory on CPU at early layers.

## [1.0.0] - 2026-09-20

### Added
- Public release of the Visual-TCAV source code on GitHub.
- Documentation and examples for external users.

### Changed
- Updated project metadata and authorship information.

## [Unreleased] - 2026-07-12

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