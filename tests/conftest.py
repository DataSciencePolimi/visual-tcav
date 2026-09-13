"""
conftest.py
-----------
Shared pytest fixtures for all visual-tcav tests.
"""

import os
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------------------
# Tiny CNN
# ---------------------------------------------------------------------------

class TinyCNN(nn.Module):
    """
    Minimal 2-layer CNN for testing.
    Input:  [B, 3, 32, 32]
    Output: [B, 10]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture(scope="session")
def tiny_model():
    """Small deterministic CNN. Created once per session."""
    torch.manual_seed(42)
    model = TinyCNN()
    model.eval()
    return model


@pytest.fixture(scope="session")
def fake_labels():
    """10 fake class labels matching TinyCNN output dimension."""
    return [f"class_{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# Fake image directories
# ---------------------------------------------------------------------------

def _make_fake_images(folder: str, n: int, size: tuple = (32, 32)) -> None:
    """
    Create n fake RGB PNG images directly inside folder.

    IMPORTANT: images go in folder/class_0/ subfolder because
    model_wrapper.py uses torchvision.datasets.ImageFolder which
    requires images to be in class subfolders.
    """
    class_folder = os.path.join(folder, "class_0")
    os.makedirs(class_folder, exist_ok=True)
    rng = np.random.RandomState(42)
    for i in range(n):
        arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(
            os.path.join(class_folder, f"img_{i:03d}.png")
        )


def _make_flat_images(folder: str, n: int, size: tuple = (32, 32)) -> None:
    """
    Create n fake RGB PNG images directly in folder (no subfolders).
    Used for test_images where LocalVisualTCAV loads images directly.
    """
    os.makedirs(folder, exist_ok=True)
    rng = np.random.RandomState(42)
    for i in range(n):
        arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(
            os.path.join(folder, f"img_{i:03d}.png")
        )


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """
    Temporary directory with all folder structure needed for tests.

    concept_images/striped/class_0/   <- images for ImageFolder
    concept_images/dotted/class_0/
    concept_images/random/class_0/
    test_images/img_000.png           <- flat images for LocalVisualTCAV
    test_images/zebra/img_*.png       <- flat images for GlobalVisualTCAV
    """
    base = tmp_path_factory.mktemp("data")

    # Concept and random folders use class_0 subfolder (ImageFolder format)
    _make_fake_images(str(base / "concept_images" / "striped"), n=5)
    _make_fake_images(str(base / "concept_images" / "dotted"),  n=5)
    _make_fake_images(str(base / "concept_images" / "random"),  n=10)

    # Test image folders use flat format (PIL.Image.open directly)
    _make_flat_images(str(base / "test_images"), n=1)
    _make_flat_images(str(base / "test_images" / "zebra"), n=3)

    return str(base)


@pytest.fixture
def cache_dir(tmp_path):
    """Fresh empty cache for each test. Uses yield for automatic cleanup."""
    c = tmp_path / "cache"
    c.mkdir()
    yield str(c)


# ---------------------------------------------------------------------------
# Pre-built TorchModelWrapper
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_wrapper(tiny_model, fake_labels):
    """
    A TorchModelWrapper around TinyCNN.

    IMPORTANT: model_preprocess must NOT include ToTensor() or
    Normalize() here. The _load_test_image() method in LocalVisualTCAV
    calls transforms.ToTensor() internally before applying model_preprocess.
    Adding ToTensor() here would apply it twice, crashing with:
    "TypeError: pic should be PIL Image or ndarray. Got torch.Tensor"
    """
    from torchvision import transforms
    from visual_tcav.model_wrapper import TorchModelWrapper

    # Only normalization — ToTensor is handled by _load_test_image
    preprocess = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    return TorchModelWrapper(
        model_name="tiny_cnn",
        model=tiny_model,
        labels=fake_labels,
        model_preprocess=preprocess,
        input_size=(3, 32, 32),
    )