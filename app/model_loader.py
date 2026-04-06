"""Model loading utilities for the house-segmentation inference service."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "best_model.pth"
_MODEL_CACHE: Dict[Tuple[str, str], nn.Module] = {}


class DoubleConv(nn.Module):
    """Two convolution layers with batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double-convolution block."""
        return self.block(x)


class UNet(nn.Module):
    """A lightweight U-Net matching the training architecture."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.encoder4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(base_channels * 2, base_channels)

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _align_tensor(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Resize decoder features when needed to match encoder shapes."""
        if source.shape[-2:] != target.shape[-2:]:
            return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return 1-channel logits."""
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self._align_tensor(self.up4(bottleneck), enc4)
        dec4 = self.decoder4(torch.cat([dec4, enc4], dim=1))

        dec3 = self._align_tensor(self.up3(dec4), enc3)
        dec3 = self.decoder3(torch.cat([dec3, enc3], dim=1))

        dec2 = self._align_tensor(self.up2(dec3), enc2)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))

        dec1 = self._align_tensor(self.up1(dec2), enc1)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))

        return self.head(dec1)


def get_device() -> torch.device:
    """Return the best available inference device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_state_dict(model_path: Path, device: torch.device) -> dict:
    """Load a model state dict in a way that is compatible across PyTorch versions."""
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {model_path}.")

    return state_dict


def get_model(model_path: str | Path | None = None) -> nn.Module:
    """Load and cache the trained U-Net model for inference."""
    resolved_model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not resolved_model_path.is_absolute():
        resolved_model_path = (PROJECT_ROOT / resolved_model_path).resolve()

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {resolved_model_path}")

    device = get_device()
    cache_key = (str(resolved_model_path), str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    state_dict = _load_state_dict(resolved_model_path, device)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load model weights. Ensure the inference U-Net matches the training architecture. "
            f"Original error: {exc}"
        ) from exc

    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


__all__ = ["DEFAULT_MODEL_PATH", "UNet", "get_device", "get_model"]
