"""
text_to_concept.py
------------------
TextToConcept: generates Concept Activation Vectors (CAVs) from plain
text descriptions using CLIP and a trained LinearAligner.

Pipeline:
    text -> CLIP text encoder -> CLIP vector (512-dim)
                                      |
                               LinearAligner
                                      |
                           CNN layer vector (e.g. 2048-dim)
                                      |
                                  Cav object

Original implementation by Daniele Di Santi (2025), based on:
    Moayeri et al., "Text-To-Concept (and Back) via Cross-Model Alignment",
    arXiv:2305.06386, 2023.
"""

import sys
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from .linear_aligner import LinearAligner
from .utils import Cav

sys.dont_write_bytecode = True


class TextToConcept:
    """
    Generates CAVs from plain text using CLIP and a LinearAligner.

    Requires the optional 'clip' dependency:
        pip install git+https://github.com/openai/CLIP.git

    Parameters
    ----------
    model_wrapper : TorchModelWrapper
        The wrapped PyTorch model. Used to extract CNN representations
        when training a new LinearAligner.
    clip_model_type : str, optional
        CLIP model variant. Default is 'ViT-B/16'.
        Other options: 'ViT-L/14', 'RN50', 'RN101'.

    Examples
    --------
    >>> from visual_tcav import TorchModelWrapper, TextToConcept
    >>> import torchvision.models as models
    >>>
    >>> resnet = models.resnet50(weights='DEFAULT')
    >>> wrapper = TorchModelWrapper(model_name="resnet50", model=resnet)
    >>>
    >>> t2c = TextToConcept(model_wrapper=wrapper)
    >>> t2c.load_aligner("./aligners/resnet50_layer4.pt")
    >>> cav = t2c.get_cav_from_text("stripes", layer_name="layer4")
    """

    def __init__(
        self,
        model_wrapper,
        clip_model_type: str = "ViT-B/16",
    ):
        self.model_wrapper = model_wrapper
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load(
                clip_model_type, device=self.device
            )
            self.clip_model.eval()
        except ImportError:
            raise ImportError(
                "The 'clip' package is required for TextToConcept. "
                "Install it with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )

        self.clip_model_type = clip_model_type
        self.linear_aligner = None

    # -----------------------------------------------------------------------
    # Aligner management
    # -----------------------------------------------------------------------

    def load_aligner(self, path: str) -> None:
        """
        Load a pre-trained LinearAligner from disk.

        Parameters
        ----------
        path : str
            Path to the saved aligner file (.pt).
        """
        self.linear_aligner = LinearAligner()
        self.linear_aligner.load_W(path)

    def save_aligner(self, path: str) -> None:
        """
        Save the current LinearAligner to disk.

        Parameters
        ----------
        path : str
            Destination path (.pt).

        Raises
        ------
        RuntimeError
            If no aligner has been trained or loaded.
        """
        self._check_aligner()
        self.linear_aligner.save_W(path)

    def train_aligner(
        self,
        dataset,
        layer_name: str,
        epochs: int = 5,
        save_path: str = None,
        cache_representations: bool = True,
        cache_dir: str = ".cache",
    ) -> None:
        """
        Train a LinearAligner on paired CNN and CLIP representations.

        Extracts CNN feature maps and CLIP image embeddings for all images
        in the dataset, then trains a linear regression to map between them.
        Representations are cached to avoid recomputation on subsequent runs.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            PyTorch dataset of images. Each item: (image, label).
        layer_name : str
            CNN layer to align (e.g. "layer4").
        epochs : int
            Training epochs. Default is 5.
        save_path : str, optional
            If provided, saves the trained aligner to this path.
        cache_representations : bool
            Cache CNN and CLIP representations to disk. Default is True.
        cache_dir : str
            Cache directory. Default is ".cache".
        """
        print("Training LinearAligner...")
        print(f"  CNN: {self.model_wrapper.model_name} @ {layer_name}")
        print(f"  CLIP: {self.clip_model_type}")

        os.makedirs(cache_dir, exist_ok=True)

        cnn_cache = os.path.join(
            cache_dir,
            f"reps_{self.model_wrapper.model_name}_{layer_name}.npy"
        )
        if cache_representations and os.path.exists(cnn_cache):
            print("  Loading CNN representations from cache...")
            cnn_reps = np.load(cnn_cache)
        else:
            print("  Extracting CNN representations...")
            cnn_reps = self._get_cnn_representations(dataset, layer_name)
            if cache_representations:
                np.save(cnn_cache, cnn_reps)

        clip_cache = os.path.join(
            cache_dir,
            f"reps_clip_{self.clip_model_type.replace('/', '_')}.npy"
        )
        if cache_representations and os.path.exists(clip_cache):
            print("  Loading CLIP representations from cache...")
            clip_reps = np.load(clip_cache)
        else:
            print("  Extracting CLIP representations...")
            clip_reps = self._get_clip_representations(dataset)
            if cache_representations:
                np.save(clip_cache, clip_reps)

        self.linear_aligner = LinearAligner()
        self.linear_aligner.train(
            source_representations=cnn_reps,
            target_representations=clip_reps,
            epochs=epochs,
        )

        if save_path is not None:
            self.linear_aligner.save_W(save_path)

    # -----------------------------------------------------------------------
    # CAV generation from text
    # -----------------------------------------------------------------------

    def get_cav_from_text(
        self,
        concept_text: str,
        layer_name: str,
        prompts: list = None,
    ) -> Cav:
        """
        Generate a CAV from a plain text concept description.

        Uses prompt ensembling (multiple templates averaged) for a more
        robust CLIP representation. The LinearAligner then translates the
        CLIP vector into the CNN's feature space to produce the CAV direction.

        Note: the returned Cav has only the direction field set.
        concept_centroid, negative_centroid, and concept_emblem are None
        because no concept images are used in this pipeline.

        Parameters
        ----------
        concept_text : str
            Text description of the concept (e.g. "stripes", "red color").
        layer_name : str
            CNN layer to generate the CAV for.
        prompts : list of str, optional
            Prompt templates. Concept text is inserted via {}.
            Default: ["a photo of {}.", "a texture of {}.", "an image of {}."]

        Returns
        -------
        Cav
            CAV with direction vector set, ready for use in Visual-TCAV.

        Raises
        ------
        RuntimeError
            If no aligner has been loaded or trained.
        """
        self._check_aligner()

        import clip as clip_lib

        if prompts is None:
            prompts = [
                "a photo of {}.",
                "a texture of {}.",
                "an image of {}.",
            ]

        # Prompt ensembling: average multiple templates for a more stable
        # concept representation (standard practice with CLIP)
        all_vectors = []
        with torch.no_grad():
            for template in prompts:
                tokens = clip_lib.tokenize([template.format(concept_text)]).to(self.device)
                vec = self.clip_model.encode_text(tokens).float()
                vec = vec / vec.norm(dim=-1, keepdim=True)
                all_vectors.append(vec)

        clip_vector = torch.stack(all_vectors).mean(dim=0)
        clip_vector = clip_vector / clip_vector.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            cav_direction = self.linear_aligner.get_aligned_representation(clip_vector)

        cav_direction = cav_direction / (cav_direction.norm(dim=-1, keepdim=True) + 1e-10)

        return Cav(direction=cav_direction.squeeze(0).detach())

    def encode_text(
        self,
        texts: list,
        prompts: list = None,
    ) -> torch.Tensor:
        """
        Encode a list of concept texts into CLIP vectors.

        Parameters
        ----------
        texts : list of str
            Concept texts to encode.
        prompts : list of str, optional
            Prompt templates. Default: ["a photo of {}."]

        Returns
        -------
        torch.Tensor
            CLIP vectors. Shape: [len(texts), clip_dim].
        """
        import clip as clip_lib

        if prompts is None:
            prompts = ["a photo of {}."]

        all_vecs = []
        with torch.no_grad():
            for text in texts:
                tokens = clip_lib.tokenize(
                    [p.format(text) for p in prompts]
                ).to(self.device)
                vecs = self.clip_model.encode_text(tokens).float()
                vec = vecs.mean(dim=0)
                vec = vec / vec.norm(dim=-1, keepdim=True)
                all_vecs.append(vec)

        return torch.stack(all_vecs)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_cnn_representations(
        self,
        dataset,
        layer_name: str,
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Extract CNN feature representations for all images in a dataset.

        GAP reduces [B, C, H, W] to [B, C] so each image is a single vector.
        """
        import torch.nn.functional as F

        # num_workers=0 required for Windows compatibility
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        all_reps = []

        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc="CNN representations"):
                feature_maps = self.model_wrapper.get_feature_maps(
                    imgs.to(self.device), layer_name
                )
                pooled = F.adaptive_avg_pool2d(feature_maps, (1, 1))
                pooled = pooled.squeeze(-1).squeeze(-1)
                all_reps.append(pooled.detach().cpu().numpy())

        return np.vstack(all_reps)

    def _get_clip_representations(
        self,
        dataset,
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Extract CLIP image representations for all images in a dataset.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        all_reps = []

        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc="CLIP representations"):
                reps = self.clip_model.encode_image(imgs.to(self.device)).float()
                reps = reps / reps.norm(dim=-1, keepdim=True)
                all_reps.append(reps.detach().cpu().numpy())

        return np.vstack(all_reps)

    def _check_aligner(self) -> None:
        if self.linear_aligner is None:
            raise RuntimeError(
                "No LinearAligner loaded. Call load_aligner(path) to load "
                "a pre-trained aligner, or train_aligner(dataset, layer) "
                "to train one from scratch."
            )

    def __repr__(self) -> str:
        aligner_status = "aligner loaded" if self.linear_aligner is not None else "no aligner"
        return (
            f"TextToConcept("
            f"model={self.model_wrapper.model_name}, "
            f"clip={self.clip_model_type}, "
            f"{aligner_status})"
        )

    def __str__(self) -> str:
        return self.__repr__()