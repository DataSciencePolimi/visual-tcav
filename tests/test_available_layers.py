"""
test_available_layers.py
------------------------
Tests for the available_layers() standalone utility function.

This is a public API function and must work correctly with:
- model name strings
- nn.Module objects
- unknown/custom models (graceful fallback)
"""

import pytest
import torch.nn as nn
from visual_tcav import available_layers


class TestAvailableLayersString:
    """Tests for available_layers() called with model name strings."""

    @pytest.mark.parametrize("model_name,expected_layer", [
        ("resnet50", "layer4"),
        ("resnet18", "layer4"),
        ("resnet101", "layer4"),
        ("vgg16", "features"),
        ("vgg19", "features"),
    ])
    def test_supported_models_print_layers(self, model_name, expected_layer, capsys):
        """available_layers() prints a table containing the expected layer names."""
        available_layers(model_name)
        captured = capsys.readouterr()
        assert expected_layer in captured.out

    @pytest.mark.parametrize("model_name", [
        "resnet50", "resnet18", "vgg16",
    ])
    def test_model_name_in_output(self, model_name, capsys):
        """The model name appears in the printed table."""
        available_layers(model_name)
        captured = capsys.readouterr()
        assert model_name in captured.out.lower()

    def test_unknown_string_raises_value_error(self):
        """available_layers raises ValueError for unsupported model name strings."""
        with pytest.raises(ValueError, match="not supported"):
            available_layers("unknown_model_xyz_123")


class TestAvailableLayersModule:
    """Tests for available_layers() called with nn.Module objects."""

    def test_known_module_with_model_name(self, capsys):
        """available_layers works with nn.Module + model_name for known models."""
        import torchvision.models as models
        model = models.resnet18(weights=None)
        available_layers(model, model_name="resnet18")
        captured = capsys.readouterr()
        assert "layer4" in captured.out

    def test_unknown_module_does_not_crash(self, capsys, tiny_model):
        """
        available_layers falls back gracefully for unknown nn.Module.
        It should print something without raising, even without labels.
        """
        available_layers(tiny_model)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_invalid_type_raises(self):
        """available_layers raises for inputs that are not str or nn.Module."""
        with pytest.raises((ValueError, AttributeError)):
            available_layers(42)