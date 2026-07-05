"""
text_to_concept.py
------------------
TextToConcept: generates Concept Activation Vectors (CAVs) from plain
text descriptions instead of requiring concept images.

Uses CLIP to encode text into a vector, then applies a trained
LinearAligner to translate that vector into the CNN's feature space,
producing a CAV direction that can be used directly in Visual-TCAV.

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

from visual_tcav.linear_aligner import LinearAligner
from visual_tcav.utils import Cav

sys.dont_write_bytecode = True


class TextToConcept:
    """
    Generates CAVs from plain text using CLIP and a LinearAligner.

    Instead of requiring 50 concept images to compute a CAV, this class
    lets you describe a concept in plain text (e.g. "stripes") and
    automatically generates the corresponding CAV vector.

    Pipeline:
        text → CLIP text encoder → CLIP vector (512-dim)
                                        ↓
                                  LinearAligner
                                        ↓
                               CNN layer vector (e.g. 2048-dim for layer4)
                                        ↓
                                   Cav object
                             (ready to use in Visual-TCAV)

    Parameters
    ----------
    model_wrapper : TorchModelWrapper
        The wrapped PyTorch model. Used to extract CNN representations
        for training the LinearAligner.
    clip_model_type : str, optional
        Which CLIP model to use. Default is 'ViT-B/16'.
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
    >>>
    >>> cav = t2c.get_cav_from_text("stripes", layer_name="layer4")
    >>> # cav is now a Cav object ready to use in LocalVisualTCAV
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

        # Load CLIP model
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load(
                clip_model_type, device=self.device
            )
            self.clip_model.eval()
        except ImportError:
            raise ImportError(
                "The 'clip' package is required for TextToConcept. "
                "Install it with: pip install git+https://github.com/openai/CLIP.git"
            )

        self.clip_model_type = clip_model_type
        self.linear_aligner = None   # loaded or trained separately

        # Cache for dataset representations (avoids recomputing)
        self._rep_cache = {}

    # -----------------------------------------------------------------------
    # Aligner management
    # -----------------------------------------------------------------------

    def load_aligner(self, path: str) -> None:
        """
        Load a pre-trained LinearAligner from disk.

        The aligner must have been trained to map from the CNN's layer
        feature space to CLIP's embedding space (or vice versa).

        Parameters
        ----------
        path : str
            Path to the saved aligner file (.pt).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        Examples
        --------
        >>> t2c.load_aligner("./aligners/resnet50_layer4.pt")
        """
        self.linear_aligner = LinearAligner()
        self.linear_aligner.load_W(path)

    def save_aligner(self, path: str) -> None:
        """
        Save the current LinearAligner to disk.

        Parameters
        ----------
        path : str
            Path where the aligner will be saved (.pt).

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
        Train a new LinearAligner on a dataset.

        Extracts both CNN and CLIP representations for all images in
        the dataset, then trains a linear regression to map between them.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            A PyTorch dataset of images. Each item should be (image, label).
        layer_name : str
            CNN layer to align with (e.g. "layer4").
        epochs : int
            Number of training epochs. Default is 5.
        save_path : str, optional
            If provided, saves the trained aligner to this path.
        cache_representations : bool
            If True, caches CNN and CLIP representations to disk.
            Default is True.
        cache_dir : str
            Directory for caching. Default is ".cache".

        Examples
        --------
        >>> from torchvision.datasets import ImageFolder
        >>> dataset = ImageFolder("./imagenet_subset")
        >>> t2c.train_aligner(
        ...     dataset=dataset,
        ...     layer_name="layer4",
        ...     epochs=5,
        ...     save_path="./aligners/resnet50_layer4.pt",
        ... )
        """
        print("Training LinearAligner...")
        print(f"  CNN model: {self.model_wrapper.model_name}, layer: {layer_name}")
        print(f"  CLIP model: {self.clip_model_type}")

        os.makedirs(cache_dir, exist_ok=True)

        # Get CNN representations
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

        # Get CLIP representations
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

        # Train the aligner: CNN → CLIP direction
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

        Encodes the text with CLIP, then uses the LinearAligner to
        translate the CLIP vector into the CNN's feature space,
        producing a CAV direction.

        Parameters
        ----------
        concept_text : str
            A text description of the concept (e.g. "stripes", "red color",
            "polka dots").
        layer_name : str
            The CNN layer to generate the CAV for.
        prompts : list of str, optional
            Template prompts for the concept. The concept text is inserted
            into each template using {}.
            Default: ["a photo of {}.", "a texture of {}.", "an image of {}."]
            Using multiple prompts and averaging improves robustness.

        Returns
        -------
        Cav
            A Cav object with the direction vector set. The concept_centroid,
            negative_centroid, and concept_emblem are None (not computed
            from images). The direction vector is ready to use in Visual-TCAV.

        Raises
        ------
        RuntimeError
            If no aligner has been loaded or trained.

        Examples
        --------
        >>> cav = t2c.get_cav_from_text("stripes", layer_name="layer4")
        >>> print(cav)
        Cav | direction from text: "stripes"
        """
        self._check_aligner()

        import clip as clip_lib

        if prompts is None:
            prompts = [
                "a photo of {}.",
                "a texture of {}.",
                "an image of {}.",
            ]

        print(f"Generating CAV from text: '{concept_text}'")

        # Step 1: Encode text with CLIP
        # Use multiple prompt templates and average for robustness
        all_vectors = []
        with torch.no_grad():
            for prompt_template in prompts:
                prompt = prompt_template.format(concept_text)
                tokens = clip_lib.tokenize([prompt]).to(self.device)
                vec = self.clip_model.encode_text(tokens)
                vec = vec.float()
                vec = vec / vec.norm(dim=-1, keepdim=True)  # normalize
                all_vectors.append(vec)

        # Average across all prompt templates
        clip_vector = torch.stack(all_vectors).mean(dim=0)
        clip_vector = clip_vector / clip_vector.norm(dim=-1, keepdim=True)

        # Step 2: Translate CLIP vector → CNN layer space
        with torch.no_grad():
            cav_direction = self.linear_aligner.get_aligned_representation(
                clip_vector
            )

        # Step 3: Normalize the direction vector
        cav_direction = cav_direction / (
            cav_direction.norm(dim=-1, keepdim=True) + 1e-10
        )

        # Step 4: Build and return a Cav object
        # Note: concept_centroid, negative_centroid, concept_emblem are None
        # because we generated the CAV from text, not from images
        cav = Cav(
            direction=cav_direction.squeeze(0).detach(),
        )

        return cav

    def encode_text(
        self,
        texts: list,
        prompts: list = None,
    ) -> torch.Tensor:
        """
        Encode a list of text concepts into CLIP vectors.

        Parameters
        ----------
        texts : list of str
            List of concept texts to encode.
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
                text_prompts = [p.format(text) for p in prompts]
                tokens = clip_lib.tokenize(text_prompts).to(self.device)
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

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            PyTorch dataset of images.
        layer_name : str
            CNN layer to extract features from.
        batch_size : int
            Batch size for processing. Default 16.

        Returns
        -------
        np.ndarray
            Feature representations. Shape: [N, feature_dim].
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_reps = []
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc="CNN representations"):
                imgs = imgs.to(self.device)
                feature_maps = self.model_wrapper.get_feature_maps(
                    imgs, layer_name
                )
                # Global average pool: [B, C, H, W] → [B, C]
                import torch.nn.functional as F
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

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            PyTorch dataset of images.
        batch_size : int
            Batch size for processing. Default 16.

        Returns
        -------
        np.ndarray
            CLIP representations. Shape: [N, clip_dim].
        """
        from torchvision import transforms

        # CLIP requires its own preprocessing
        clip_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_reps = []
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc="CLIP representations"):
                imgs = imgs.to(self.device)
                reps = self.clip_model.encode_image(imgs).float()
                reps = reps / reps.norm(dim=-1, keepdim=True)
                all_reps.append(reps.detach().cpu().numpy())

        return np.vstack(all_reps)

    def _check_aligner(self) -> None:
        """Raise RuntimeError if no aligner has been loaded or trained."""
        if self.linear_aligner is None:
            raise RuntimeError(
                "No LinearAligner loaded. "
                "Call load_aligner(path) to load a pre-trained aligner, "
                "or call train_aligner(dataset, layer_name) to train one."
            )

    def __repr__(self) -> str:
        aligner_status = (
            "aligner loaded" if self.linear_aligner is not None
            else "no aligner"
        )
        return (
            f"TextToConcept("
            f"model={self.model_wrapper.model_name}, "
            f"clip={self.clip_model_type}, "
            f"{aligner_status})"
        )

    def __str__(self) -> str:
        return self.__repr__()