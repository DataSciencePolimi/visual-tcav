"""
linear_aligner.py
-----------------
Linear Aligner: maps representations from one embedding space to another
via a learned linear transformation (y = x @ W.T + b).

Used in the Text-to-Concept pipeline to translate CLIP text embeddings
(512-dim) into CNN feature space (e.g. 2048-dim for ResNet50 layer4),
enabling CAV generation from plain text descriptions.

Original implementation by Daniele Di Santi (2025), based on:
    Moayeri et al., "Text-To-Concept (and Back) via Cross-Model Alignment",
    arXiv:2305.06386, 2023.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.dont_write_bytecode = True


class _LinearRegression(nn.Module):
    """
    Single-layer linear model: y = W @ x + b.

    Used internally by _LinearRegressionSolver.
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _LinearRegressionSolver:
    """
    Trains a _LinearRegression model to map one representation space
    to another using SGD optimization.

    Both spaces are normalized to a target variance before training
    to improve numerical stability across different embedding scales.
    """

    def __init__(self):
        self.model = None
        self.criterion = nn.MSELoss()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bias: bool = True,
        batch_size: int = 100,
        epochs: int = 20,
    ) -> None:
        """
        Train the linear regression model.

        Parameters
        ----------
        X : np.ndarray
            Input representations. Shape: [N, input_dim].
        y : np.ndarray
            Target representations. Shape: [N, output_dim].
        bias : bool
            Whether to include a bias term. Default True.
        batch_size : int
            Training batch size. Default 100.
        epochs : int
            Number of training epochs. Default 20.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )
        # num_workers=0 required for Windows compatibility
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        self.model = _LinearRegression(X.shape[1], y.shape[1], bias=bias).to(device)

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
        )
        # Cosine annealing decays the learning rate smoothly to near zero
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200
        )

        self.model.train()
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(inputs), targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

    def test(self, X: np.ndarray, y: np.ndarray, batch_size: int = 100):
        """
        Evaluate the model and return MSE and R² score.

        Parameters
        ----------
        X : np.ndarray
            Input representations. Shape: [N, input_dim].
        y : np.ndarray
            Target representations. Shape: [N, output_dim].
        batch_size : int
            Evaluation batch size. Default 100.

        Returns
        -------
        tuple of (float, float)
            MSE loss and R² score.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        self.model.eval()
        total_mse, n_batches = 0.0, 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                total_mse += self.criterion(self.model(inputs), targets).item()
                n_batches += 1

        total_mse /= n_batches
        # R² = 1 means perfect fit; 0 means no better than predicting the mean
        r2 = 1 - total_mse / self.get_variance(y)
        return total_mse, r2

    def extract_parameters(self):
        """Return the learned weight matrix W and bias vector b."""
        W, b = None, None
        for name, param in self.model.named_parameters():
            if name == "linear.weight":
                W = param.detach()
            elif name == "linear.bias":
                b = param.detach()
        return W, b

    @staticmethod
    def get_variance(y: np.ndarray) -> float:
        """Compute variance as E[y²] - E[y]²."""
        return float(np.mean(np.square(y)) - np.mean(y) ** 2)


class LinearAligner:
    """
    Maps representations from one embedding space to another using a
    learned linear transformation: output = input @ W.T + b.

    In the Text-to-Concept pipeline, translates CLIP embeddings (512-dim)
    into CNN layer feature space (e.g. 2048-dim for ResNet50 layer4).

    Parameters
    ----------
    None. Call train() or load_W() before calling get_aligned_representation().

    Examples
    --------
    Training a new aligner:

    >>> aligner = LinearAligner()
    >>> aligner.train(cnn_reps, clip_reps, epochs=5)
    >>> aligner.save_W("./aligners/resnet50_layer4.pt")

    Loading a pre-trained aligner:

    >>> aligner = LinearAligner()
    >>> aligner.load_W("./aligners/resnet50_layer4.pt")
    >>> aligned = aligner.get_aligned_representation(clip_vector)
    """

    def __init__(self):
        self.W = None
        self.b = None

    def train(
        self,
        source_representations: np.ndarray,
        target_representations: np.ndarray,
        epochs: int = 5,
        target_variance: float = 4.5,
    ) -> None:
        """
        Train the linear aligner to map source → target representations.

        Both spaces are scaled to target_variance before training to equalize
        their numerical ranges (from Moayeri et al., 2023). The scaling is
        reversed after training so W and b operate on the original vectors.

        Parameters
        ----------
        source_representations : np.ndarray
            Source space representations (e.g. CNN features). Shape: [N, src_dim].
        target_representations : np.ndarray
            Target space representations (e.g. CLIP features). Shape: [N, tgt_dim].
        epochs : int
            Training epochs. Default is 5.
        target_variance : float
            Both spaces are scaled to this variance. Default is 4.5.

        Raises
        ------
        ValueError
            If source and target have different numbers of samples.
        """
        if source_representations.shape[0] != target_representations.shape[0]:
            raise ValueError(
                f"source and target must have the same number of samples, "
                f"got {source_representations.shape[0]} and "
                f"{target_representations.shape[0]}."
            )

        print(
            f"Training LinearAligner: "
            f"{source_representations.shape} -> {target_representations.shape}"
        )

        solver = _LinearRegressionSolver()

        var_source = solver.get_variance(source_representations)
        var_target = solver.get_variance(target_representations)

        c_source = (target_variance / var_source) ** 0.5
        c_target = (target_variance / var_target) ** 0.5

        solver.train(
            c_source * source_representations,
            c_target * target_representations,
            bias=True,
            epochs=epochs,
            batch_size=100,
        )

        mse, r2 = solver.test(
            c_source * source_representations,
            c_target * target_representations,
        )
        print(f"  Final MSE={mse:.3f}, R²={r2:.3f}")

        W, b = solver.extract_parameters()
        # Rescale back so W and b operate on the original (unscaled) vectors
        self.W = W * c_source / c_target
        self.b = b * c_source / c_target

    def get_aligned_representation(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Map input features from source space to target space.

        Parameters
        ----------
        features : torch.Tensor
            Input vectors. Shape: [N, source_dim].

        Returns
        -------
        torch.Tensor
            Aligned vectors. Shape: [N, target_dim].

        Raises
        ------
        RuntimeError
            If the aligner has not been trained or loaded.
        """
        if self.W is None or self.b is None:
            raise RuntimeError(
                "LinearAligner not trained. Call train() or load_W() first."
            )
        return features @ self.W.T + self.b

    def load_W(self, path: str) -> None:
        """
        Load a pre-trained aligner from disk.

        Parameters
        ----------
        path : str
            Path to the saved .pt file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Aligner file not found at: {path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        aligner_dict = torch.load(path, weights_only=False, map_location=device)

        self.W = aligner_dict["W"].float().to(device)
        self.b = aligner_dict["b"].float().to(device)

        print(f"LinearAligner loaded: W={self.W.shape}, b={self.b.shape}")

    def save_W(self, path: str) -> None:
        """
        Save the trained aligner to disk.

        Parameters
        ----------
        path : str
            Destination path (.pt).

        Raises
        ------
        RuntimeError
            If the aligner has not been trained.
        """
        if self.W is None or self.b is None:
            raise RuntimeError(
                "Cannot save: LinearAligner not trained. Call train() first."
            )

        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True,
        )
        torch.save({"W": self.W.detach().cpu(), "b": self.b.detach().cpu()}, path)
        print(f"LinearAligner saved to: {path}")

    def __repr__(self) -> str:
        if self.W is not None:
            return (
                f"LinearAligner("
                f"source_dim={self.W.shape[1]}, "
                f"target_dim={self.W.shape[0]})"
            )
        return "LinearAligner(untrained)"

    def __str__(self) -> str:
        return self.__repr__()