"""
test_utils.py
-------------
Unit tests for visual_tcav/utils.py.
"""

import pytest
import torch
from visual_tcav.utils import Cav, ConceptLayer, Stat, Predictions, Prediction


class TestCav:
    """Tests for the Cav data class."""

    def test_creation_with_direction(self):
        """A Cav can be created with just a direction tensor."""
        direction = torch.randn(512)
        cav = Cav(direction=direction)
        assert cav.direction is not None
        assert cav.direction.shape == (512,)

    def test_creation_empty(self):
        """A Cav created with no arguments has all fields as None."""
        cav = Cav()
        assert cav.direction is None
        assert cav.concept_centroid is None
        assert cav.negative_centroid is None
        assert cav.concept_emblem is None

    def test_cpu_moves_all_tensors(self):
        """Cav.cpu() moves all tensors to CPU."""
        cav = Cav(
            direction=torch.randn(64),
            concept_emblem=torch.randn(64),
        )
        cav_cpu = cav.cpu()
        assert cav_cpu.direction.device.type == "cpu"
        assert cav_cpu.concept_emblem.device.type == "cpu"

    def test_to_device(self):
        """Cav.to(device) moves all tensors to the specified device."""
        cav = Cav(direction=torch.randn(64))
        cav = cav.to(torch.device("cpu"))
        assert cav.direction.device.type == "cpu"

    @pytest.mark.parametrize("size", [32, 64, 128, 512, 2048])
    def test_direction_various_channel_sizes(self, size):
        """Cav stores direction tensors for any valid CNN channel count."""
        cav = Cav(direction=torch.randn(size))
        assert cav.direction.shape == (size,)

    def test_str_representation(self):
        """Cav has a readable string representation."""
        assert "Cav" in str(Cav())


class TestStat:
    """Tests for the Stat class used by GlobalVisualTCAV."""

    @pytest.mark.parametrize("scores,expected_mean", [
        ([1.0, 2.0, 3.0, 4.0, 5.0], 3.0),
        ([0.0, 0.0, 0.0], 0.0),
        ([10.0, 10.0], 10.0),
        ([0.1, 0.2, 0.3], pytest.approx(0.2, abs=1e-4)),
    ])
    def test_mean(self, scores, expected_mean):
        """Mean is computed correctly for various score distributions."""
        stat = Stat(scores)
        assert stat.mean.item() == pytest.approx(expected_mean, abs=1e-4)

    def test_std_unbiased(self):
        """
        Standard deviation uses correction=1 (unbiased estimator, ddof=1).
        Uses [0.0, 2.0, 4.0] where the exact result is 2.0 for both
        numpy ddof=1 and torch correction=1.
        """
        scores = [0.0, 2.0, 4.0]
        stat = Stat(scores)
        assert stat.std.item() == pytest.approx(2.0, abs=1e-4)

    def test_std_is_non_negative(self):
        """Standard deviation is always non-negative."""
        stat = Stat([1.0, 2.0, 3.0, 4.0])
        assert stat.std.item() >= 0.0

    def test_confidence_interval_lower_clipped_to_zero(self):
        """
        Lower CI bound is never negative.
        Attribution scores are non-negative by design (ReLU applied),
        so the CI lower bound is clipped to 0 using relu.
        """
        stat = Stat([0.01, 0.01, 0.01, 0.01])
        assert stat.begin.item() >= 0.0

    def test_confidence_interval_upper(self):
        """Upper CI bound equals mean + 2 * std_err."""
        scores = [1.0, 2.0, 3.0]
        stat = Stat(scores)
        expected = stat.mean.item() + 2.0 * stat.std_err.item()
        assert stat.end.item() == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("n", [2, 5, 10, 50])
    def test_n_stored_correctly(self, n):
        """Number of samples is stored correctly for any list length."""
        stat = Stat([0.5] * n)
        assert stat.n == n

    def test_two_values_no_crash(self):
        """Stat handles two values without crashing (minimum for ddof=1)."""
        stat = Stat([1.0, 3.0])
        assert abs(stat.mean.item() - 2.0) < 1e-5


class TestConceptLayer:
    """Tests for the ConceptLayer data class."""

    def test_default_initialization(self):
        """ConceptLayer initializes with no concept map and empty attributions."""
        cl = ConceptLayer()
        assert cl.concept_map is None
        assert cl.attributions == {}

    def test_stores_cav(self):
        """ConceptLayer stores a CAV correctly."""
        cav = Cav(direction=torch.randn(64))
        cl = ConceptLayer(cav=cav)
        assert cl.cav.direction is not None

    @pytest.mark.parametrize("class_index,score", [
        (0, 0.42),
        (1, 0.0),
        (999, 1.0),
    ])
    def test_stores_attribution_by_class_index(self, class_index, score):
        """Attribution scores are stored and retrieved by class index."""
        cl = ConceptLayer()
        cl.attributions[class_index] = torch.tensor(score)
        assert cl.attributions[class_index].item() == pytest.approx(score, abs=1e-6)


class TestPredictions:
    """Tests for Prediction and Predictions data classes."""

    @pytest.mark.parametrize("path,expected_filename", [
        ("/data/test/cat.jpg", "cat.jpg"),
        ("/home/user/zebra.png", "zebra.png"),
        ("relative/path/img.jpeg", "img.jpeg"),
    ])
    def test_filename_extraction(self, path, expected_filename):
        """Predictions extracts the filename from any path format."""
        preds = Predictions(
            predictions=[[Prediction("cat", 0, 0.9)]],
            test_image_path=path,
            model_name="resnet50",
        )
        assert preds.test_image_filename == expected_filename

    def test_getitem(self):
        """Predictions supports list-style indexing."""
        inner = [Prediction("cat", 0, 0.9)]
        preds = Predictions(
            predictions=[inner],
            test_image_path="cat.jpg",
            model_name="resnet50",
        )
        assert preds[0] == inner

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0, 0.99])
    def test_prediction_confidence_stored(self, confidence):
        """Prediction stores confidence values correctly."""
        p = Prediction(class_name="zebra", class_index=340, confidence=confidence)
        assert p.confidence == pytest.approx(confidence, abs=1e-7)