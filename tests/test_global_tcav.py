"""
test_global_tcav.py
-------------------
Tests for GlobalVisualTCAV.

Mirrors the structure of test_local_tcav.py.
Uses the tiny_wrapper and data_dir fixtures from conftest.py.
"""

import os
import pytest
from visual_tcav import GlobalVisualTCAV


class TestGlobalVisualTCAVInstantiation:
    """Tests for constructor behavior."""

    def test_instantiation_with_wrapper(self, tiny_wrapper, data_dir, cache_dir):
        """GlobalVisualTCAV can be created with a pre-built wrapper."""
        tcav = GlobalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_images_dir=os.path.join(data_dir, "test_images", "zebra"),
            concept_names=["striped"],
            concept_base_dir=os.path.join(data_dir, "concept_images"),
            random_dir=os.path.join(data_dir, "concept_images", "random"),
            layer_names=["conv1"],
            cache_dir=cache_dir,
        )
        assert tcav.model_wrapper is not None
        assert len(tcav.test_image_paths) == 3

    def test_invalid_images_dir_raises(self, tiny_wrapper, cache_dir):
        """FileNotFoundError if test_images_dir does not exist."""
        with pytest.raises(FileNotFoundError):
            GlobalVisualTCAV(
                model_wrapper=tiny_wrapper,
                test_images_dir="/nonexistent/folder",
                cache_dir=cache_dir,
            )

    def test_empty_images_dir_raises(self, tiny_wrapper, tmp_path, cache_dir):
        """ValueError if test_images_dir contains no valid images."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No valid images"):
            GlobalVisualTCAV(
                model_wrapper=tiny_wrapper,
                test_images_dir=str(empty),
                cache_dir=cache_dir,
            )

    @pytest.mark.parametrize("max_images,expected_count", [
        (1, 1),
        (2, 2),
        (3, 3),
        (100, 3),  # only 3 images exist, cap is irrelevant
    ])
    def test_max_test_images_respected(
        self, tiny_wrapper, data_dir, cache_dir, max_images, expected_count
    ):
        """max_test_images limits the number of images actually loaded."""
        tcav = GlobalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_images_dir=os.path.join(data_dir, "test_images", "zebra"),
            max_test_images=max_images,
            cache_dir=cache_dir,
        )
        assert len(tcav.test_image_paths) == expected_count


class TestGlobalVisualTCAVExplain:
    """Tests for the explain() method."""

    @pytest.fixture
    def configured_tcav(self, tiny_wrapper, data_dir, cache_dir):
        """A fully configured GlobalVisualTCAV ready to explain."""
        return GlobalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_images_dir=os.path.join(data_dir, "test_images", "zebra"),
            concept_names=["striped"],
            concept_base_dir=os.path.join(data_dir, "concept_images"),
            random_dir=os.path.join(data_dir, "concept_images", "random"),
            layer_names=["conv1"],
            max_test_images=3,
            cache_dir=cache_dir,
        )

    def test_explain_runs_without_error(self, configured_tcav):
        """explain() completes without raising."""
        configured_tcav.explain()
        assert configured_tcav.stats != {}

    def test_stats_contain_stat_objects(self, configured_tcav):
        """After explain(), stats contains Stat objects with correct attributes."""
        configured_tcav.explain()
        stat = configured_tcav.stats["conv1"]["striped"][0]
        assert hasattr(stat, "mean")
        assert hasattr(stat, "std")
        assert hasattr(stat, "begin")
        assert hasattr(stat, "end")
        assert hasattr(stat, "n")

    def test_stats_mean_is_non_negative(self, configured_tcav):
        """Attribution scores are non-negative so mean must be non-negative."""
        configured_tcav.explain()
        stat = configured_tcav.stats["conv1"]["striped"][0]
        assert stat.mean.item() >= 0.0

    def test_explain_missing_images_raises(self, tiny_wrapper, cache_dir):
        """explain() raises RuntimeError if no test images configured."""
        tcav = GlobalVisualTCAV(model_wrapper=tiny_wrapper, cache_dir=cache_dir)
        with pytest.raises(RuntimeError, match="test_images_dir"):
            tcav.explain()

    def test_explain_missing_concepts_raises(self, tiny_wrapper, data_dir, cache_dir):
        """explain() raises RuntimeError if no concepts configured."""
        tcav = GlobalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_images_dir=os.path.join(data_dir, "test_images", "zebra"),
            layer_names=["conv1"],
            cache_dir=cache_dir,
        )
        with pytest.raises(RuntimeError, match="concept_names"):
            tcav.explain()

    def test_explain_missing_layers_raises(self, tiny_wrapper, data_dir, cache_dir):
        """explain() raises RuntimeError if no layers configured."""
        tcav = GlobalVisualTCAV(
            model_wrapper=tiny_wrapper,
            test_images_dir=os.path.join(data_dir, "test_images", "zebra"),
            concept_names=["striped"],
            concept_base_dir=os.path.join(data_dir, "concept_images"),
            random_dir=os.path.join(data_dir, "concept_images", "random"),
            cache_dir=cache_dir,
        )
        with pytest.raises(RuntimeError, match="layer_names"):
            tcav.explain()

    def test_force_recompute_still_produces_results(self, configured_tcav):
        """explain(force_recompute=True) ignores cache and results remain valid."""
        configured_tcav.explain()
        configured_tcav.explain(force_recompute=True)
        assert configured_tcav.stats["conv1"]["striped"][0].mean is not None

    def test_stats_info_before_explain_raises(self, configured_tcav):
        """statsInfo() raises RuntimeError if explain() has not been called."""
        with pytest.raises(RuntimeError, match="explain"):
            configured_tcav.statsInfo()