"""Generate prediction visualizations for the trained house segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from training.dataset import HouseSegmentationDataset
except ImportError:
    from dataset import HouseSegmentationDataset


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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for visualization generation."""
    parser = argparse.ArgumentParser(description="Generate segmentation prediction visualizations.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of test samples to visualize.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square resize dimension used by the dataset.",
    )
    parser.add_argument(
        "--test-images-dir",
        type=Path,
        default=Path("data/test/images"),
        help="Path to test images.",
    )
    parser.add_argument(
        "--test-masks-dir",
        type=Path,
        default=Path("data/test/masks"),
        help="Path to test masks.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/best_model.pth"),
        help="Path to the trained model weights.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before generating visualizations."""
    if args.num_samples <= 0:
        raise ValueError(f"--num-samples must be positive, got {args.num_samples}.")
    if args.image_size <= 0:
        raise ValueError(f"--image-size must be positive, got {args.image_size}.")


def load_model(model_path: Path, device: torch.device) -> UNet:
    """Load the trained U-Net weights and return the model in eval mode."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_prediction_figure(
    image_tensor: torch.Tensor,
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    output_path: Path,
) -> None:
    """Save a three-panel figure with input image, ground truth, and prediction."""
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    true_mask_np = true_mask.detach().cpu().squeeze(0).numpy()
    pred_mask_np = pred_mask.detach().cpu().squeeze(0).numpy()

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[1].imshow(true_mask_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_mask_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Predicted Mask")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def generate_visualizations(
    model: UNet,
    dataset: HouseSegmentationDataset,
    output_dir: Path,
    num_samples: int,
    device: torch.device,
) -> None:
    """Generate and save prediction figures for the first N test samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(num_samples, len(dataset))

    for index in range(sample_count):
        image_tensor, true_mask = dataset[index]
        logits = model(image_tensor.unsqueeze(0).to(device))
        pred_mask = (torch.sigmoid(logits) >= 0.5).float().cpu().squeeze(0)

        output_path = output_dir / f"prediction_{index + 1:03d}.png"
        save_prediction_figure(
            image_tensor=image_tensor,
            true_mask=true_mask,
            pred_mask=pred_mask,
            output_path=output_path,
        )
        print(f"Saved visualization {index + 1}/{sample_count} to {output_path}")


def main() -> None:
    """Load the trained model and generate deterministic prediction examples."""
    args = parse_args()
    validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs/predictions")

    dataset = HouseSegmentationDataset(
        images_dir=args.test_images_dir,
        masks_dir=args.test_masks_dir,
        image_size=args.image_size,
        augment=False,
    )
    model = load_model(args.model_path, device=device)

    generate_visualizations(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        num_samples=args.num_samples,
        device=device,
    )


if __name__ == "__main__":
    main()
