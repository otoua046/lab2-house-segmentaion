"""Dataset loading utilities for binary house segmentation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def binarize_mask(mask: Image.Image) -> torch.Tensor:
    """Convert a mask image into a binary float tensor with values 0.0 or 1.0."""
    mask_array = np.array(mask, dtype=np.uint8)

    # Treat any non-zero pixel as foreground so slightly different mask encodings
    # still collapse cleanly into the required binary segmentation target.
    binary_mask = (mask_array > 0).astype(np.float32)
    return torch.from_numpy(binary_mask).unsqueeze(0)


class HouseSegmentationDataset(Dataset):
    """PyTorch dataset for paired RGB house images and binary masks."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = int(image_size)
        self.augment = augment

        self.image_paths, self.mask_paths = self._validate_and_collect_pairs()

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )
        self.mask_resize = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST,
        )

    def _validate_and_collect_pairs(self) -> tuple[list[Path], list[Path]]:
        """Validate directory contents and collect strictly matched image/mask pairs."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory does not exist: {self.masks_dir}")
        if not self.images_dir.is_dir():
            raise NotADirectoryError(f"Expected images_dir to be a directory: {self.images_dir}")
        if not self.masks_dir.is_dir():
            raise NotADirectoryError(f"Expected masks_dir to be a directory: {self.masks_dir}")

        image_paths = sorted(path for path in self.images_dir.iterdir() if path.is_file())
        mask_paths = sorted(path for path in self.masks_dir.iterdir() if path.is_file())

        if not image_paths:
            raise ValueError(f"No image files found in {self.images_dir}.")
        if not mask_paths:
            raise ValueError(f"No mask files found in {self.masks_dir}.")

        image_names = [path.name for path in image_paths]
        mask_names = [path.name for path in mask_paths]

        if len(image_names) != len(mask_names):
            raise ValueError(
                "Image/mask count mismatch: "
                f"{len(image_names)} images vs {len(mask_names)} masks."
            )

        if image_names != mask_names:
            missing_masks = sorted(set(image_names) - set(mask_names))
            missing_images = sorted(set(mask_names) - set(image_names))
            details: list[str] = []
            if missing_masks:
                details.append(f"missing masks for: {', '.join(missing_masks[:10])}")
            if missing_images:
                details.append(f"missing images for: {', '.join(missing_images[:10])}")
            raise ValueError(
                "Images and masks must match exactly by filename. "
                + " | ".join(details)
            )

        return image_paths, mask_paths

    def __len__(self) -> int:
        """Return the number of paired samples."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one image/mask pair and return tensors ready for segmentation."""
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")

        # Apply the same random spatial augmentations to the image and mask so
        # the segmentation target remains pixel-aligned with the input.
        if self.augment:
            if random.random() < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
            if random.random() < 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

        image_tensor = self.image_transform(image)
        resized_mask = self.mask_resize(mask)
        mask_tensor = binarize_mask(resized_mask)

        return image_tensor, mask_tensor
