"""
test_local_tcav.py
------------------
Tests for LocalVisualTCAV.

Uses the tiny_wrapper fixture from conftest.py to avoid loading
ResNet50. Tests run in seconds on CPU.

Test categories:
1. Instantiation
2. predict()
3. explain() end to end
4. Caching behavior
5. plot()
"""

import os
import pytest
import torch
from visual_tcav import LocalVisualTCAV


class TestLocalVisualTCAVInstantiation:
    """Tests for constructor behavior."""

    def test_instantiation_with_wrapper(self, tiny_wrapper, data_dir, cache_dir):
        """LocalVisualTCAV can be created with a pre-built wrapper."""
        tcav = LocalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_image_path=os.path.join(data_dir, "test_images", "img_000.png"),
            concept_names=["striped"],
            concept_base_dir=os.path.join(data_dir, "concept_images"),
            random_dir=os.path.join(data_dir, "concept_images", "random"),
            layer_names=["conv1"],
            cache_dir=cache_dir,
        )
        assert tcav.model_wrapper is not None
        assert tcav.concept_names == ["striped"]
        assert tcav.layer_names == ["conv1"]

    def test_instantiation_minimal(self, tiny_wrapper):
        """LocalVisualTCAV can be created with only a model_wrapper (all optional)."""
        tcav = LocalVisualTCAV(model_wrapper=tiny_wrapper)
        assert tcav.test_image_tensor is None
        assert tcav.concept_names == []
        assert tcav.layer_names == []

    def test_no_model_raises(self):
        """Constructor raises ValueError if neither model nor model_wrapper given."""
        with pytest.raises(ValueError):
            LocalVisualTCAV()

    def test_invalid_image_path_raises(self, tiny_wrapper):
        """FileNotFoundError if test image path does not exist."""
        with pytest.raises(FileNotFoundError):
            LocalVisualTCAV(
                model_wrapper=tiny_wrapper,
                test_image_path="/nonexistent/path/image.jpg",
            )

    def test_invalid_concept_dir_raises(self, tiny_wrapper, data_dir):
        """ValueError if concept folder does not exist."""
        with pytest.raises(ValueError):
            LocalVisualTCAV(
                model_wrapper=tiny_wrapper,
                concept_names=["nonexistent_concept"],
                concept_base_dir=os.path.join(data_dir, "concept_images"),
            )

    def test_cache_dir_none_accepted(self, tiny_wrapper):
        """cache_dir=None disables caching without raising."""
        tcav = LocalVisualTCAV(model_wrapper=tiny_wrapper, cache_dir=None)
        assert tcav.cache_dir is None

    @pytest.mark.parametrize("n_classes", [1, 3, 5])
    def test_n_classes_stored(self, tiny_wrapper, n_classes):
        """n_classes parameter is stored correctly."""
        tcav = LocalVisualTCAV(model_wrapper=tiny_wrapper, n_classes=n_classes)
        assert tcav.n_classes == n_classes


class TestLocalVisualTCAVPredict:
    """Tests for the predict() method."""

    def test_predict_returns_predictions(self, tiny_wrapper, data_dir, cache_dir):
        """predict() returns a Predictions object with n_classes predictions."""
        tcav = LocalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_image_path=os.path.join(data_dir, "test_images", "img_000.png"),
            cache_dir=cache_dir,
        )
        preds = tcav.predict()
        assert preds is not None
        assert len(preds.predictions[0]) == tcav.n_classes

    def test_predict_without_image_raises(self, tiny_wrapper, cache_dir):
        """predict() raises RuntimeError with clear message if no image set."""
        tcav = LocalVisualTCAV(model_wrapper=tiny_wrapper, cache_dir=cache_dir)
        with pytest.raises(RuntimeError, match="test_image_path"):
            tcav.predict()

    def test_predict_sets_target_classes(self, tiny_wrapper, data_dir, cache_dir):
        """predict() populates self.target_classes."""
        tcav = LocalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_image_path=os.path.join(data_dir, "test_images", "img_000.png"),
            cache_dir=cache_dir,
        )
        tcav.predict()
        assert len(tcav.target_classes) == tcav.n_classes


class TestLocalVisualTCAVExplain:
    """Tests for the explain() method."""

    @pytest.fixture
    def configured_tcav(self, tiny_wrapper, data_dir, cache_dir):
        """A fully configured LocalVisualTCAV ready to explain."""
        return LocalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_image_path=os.path.join(data_dir, "test_images", "img_000.png"),
            concept_names=["striped"],
            concept_base_dir=os.path.join(data_dir, "concept_images"),
            random_dir=os.path.join(data_dir, "concept_images", "random"),
            layer_names=["conv1"],
            cache_dir=cache_dir,
        )

    def test_explain_runs_without_predict(self, configured_tcav):
        """explain() completes without an explicit predict() call."""
        configured_tcav.explain()
        assert configured_tcav.computations["conv1"]["striped"].concept_map is not None

    def test_explain_auto_calls_predict(self, configured_tcav):
        """explain() sets target_classes even without an explicit predict() call."""
        configured_tcav.explain()
        assert len(configured_tcav.target_classes) > 0

    def test_concept_map_is_2d(self, configured_tcav):
        """Concept map is a 2D tensor (H x W)."""
        configured_tcav.explain()
        concept_map = configured_tcav.computations["conv1"]["striped"].concept_map
        assert concept_map.ndim == 2

    def test_concept_map_non_negative(self, configured_tcav):
        """Concept map values are non-negative (ReLU applied in pipeline)."""
        configured_tcav.explain()
        concept_map = configured_tcav.computations["conv1"]["striped"].concept_map
        assert (concept_map >= 0).all()

    def test_attribution_scores_exist_for_all_classes(self, configured_tcav):
        """Attribution scores are computed for each target class."""
        configured_tcav.explain()
        attributions = configured_tcav.computations["conv1"]["striped"].attributions
        assert len(attributions) == configured_tcav.n_classes

    def test_explain_missing_image_raises(self, tiny_wrapper, cache_dir):
        """explain() raises RuntimeError with actionable message if no image."""
        tcav = LocalVisualTCAV(model_wrapper=tiny_wrapper, cache_dir=cache_dir)
        with pytest.raises(RuntimeError, match="test_image_path"):
            tcav.explain()

    def test_explain_missing_concepts_raises(self, tiny_wrapper, data_dir, cache_dir):
        """explain() raises RuntimeError with actionable message if no concepts."""
        tcav = LocalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_image_path=os.path.join(data_dir, "test_images", "img_000.png"),
            layer_names=["conv1"],
            cache_dir=cache_dir,
        )
        with pytest.raises(RuntimeError, match="concept_names"):
            tcav.explain()

    def test_explain_missing_layers_raises(self, tiny_wrapper, data_dir, cache_dir):
        """explain() raises RuntimeError with actionable message if no layers."""
        tcav = LocalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_image_path=os.path.join(data_dir, "test_images", "img_000.png"),
            concept_names=["striped"],
            concept_base_dir=os.path.join(data_dir, "concept_images"),
            random_dir=os.path.join(data_dir, "concept_images", "random"),
            cache_dir=cache_dir,
        )
        with pytest.raises(RuntimeError, match="layer_names"):
            tcav.explain()


class TestLocalVisualTCAVCache:
    """Tests for the caching system."""

    @pytest.fixture
    def full_tcav(self, tiny_wrapper, data_dir, cache_dir):
        return LocalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_image_path=os.path.join(data_dir, "test_images", "img_000.png"),
            concept_names=["striped"],
            concept_base_dir=os.path.join(data_dir, "concept_images"),
            random_dir=os.path.join(data_dir, "concept_images", "random"),
            layer_names=["conv1"],
            cache_dir=cache_dir,
        )

    def test_cache_files_created_after_explain(self, full_tcav, cache_dir):
        """explain() creates .joblib files in cache_dir."""
        full_tcav.explain()
        joblib_files = [f for f in os.listdir(cache_dir) if f.endswith(".joblib")]
        assert len(joblib_files) > 0

    def test_clear_cache_removes_all_files(self, full_tcav, cache_dir):
        """clear_cache() deletes all cached files and recreates empty dir."""
        full_tcav.explain()
        full_tcav.clear_cache()
        assert os.path.exists(cache_dir)
        assert len(os.listdir(cache_dir)) == 0

    def test_clear_cache_none_does_not_crash(self, tiny_wrapper):
        """clear_cache() with cache_dir=None prints a message without crashing."""
        tcav = LocalVisualTCAV(model_wrapper=tiny_wrapper, cache_dir=None)
        tcav.clear_cache()  # must not raise

    def test_force_recompute_produces_valid_results(self, full_tcav):
        """explain(force_recompute=True) ignores cache and still produces results."""
        full_tcav.explain()
        full_tcav.explain(force_recompute=True)
        assert full_tcav.computations["conv1"]["striped"].concept_map is not None

    def test_cache_dir_none_writes_no_files(self, tiny_wrapper, data_dir, tmp_path):
        """With cache_dir=None no .joblib files are written anywhere."""
        tcav = LocalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_image_path=os.path.join(data_dir, "test_images", "img_000.png"),
            concept_names=["striped"],
            concept_base_dir=os.path.join(data_dir, "concept_images"),
            random_dir=os.path.join(data_dir, "concept_images", "random"),
            layer_names=["conv1"],
            cache_dir=None,
        )
        tcav.explain()
        assert len(list(tmp_path.rglob("*.joblib"))) == 0


class TestLocalVisualTCAVPlot:
    """Tests for the plot() method."""

    def test_plot_before_explain_raises(self, tiny_wrapper, data_dir, cache_dir):
        """plot() raises RuntimeError if explain() has not been called yet."""
        tcav = LocalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_image_path=os.path.join(data_dir, "test_images", "img_000.png"),
            concept_names=["striped"],
            concept_base_dir=os.path.join(data_dir, "concept_images"),
            random_dir=os.path.join(data_dir, "concept_images", "random"),
            layer_names=["conv1"],
            cache_dir=cache_dir,
        )
        with pytest.raises(RuntimeError, match="explain"):
            tcav.plot()

    def test_plot_saves_png(self, tiny_wrapper, data_dir, cache_dir, tmp_path):
        """plot(save_path=...) saves a non-empty PNG file."""
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — no display needed

        tcav = LocalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_image_path=os.path.join(data_dir, "test_images", "img_000.png"),
            concept_names=["striped"],
            concept_base_dir=os.path.join(data_dir, "concept_images"),
            random_dir=os.path.join(data_dir, "concept_images", "random"),
            layer_names=["conv1"],
            cache_dir=cache_dir,
        )
        tcav.explain()
        save_path = str(tmp_path / "output.png")
        tcav.plot(save_path=save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0