"""Evaluate a trained U-Net on the house segmentation test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from training.dataset import HouseSegmentationDataset
    from training.metrics import binary_dice, binary_iou
except ImportError:
    from dataset import HouseSegmentationDataset
    from metrics import binary_dice, binary_iou


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


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute Dice loss from logits and binary masks."""
    probabilities = torch.sigmoid(logits).flatten(start_dim=1)
    targets = targets.float().flatten(start_dim=1)

    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice_score.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor, bce_loss: nn.Module) -> torch.Tensor:
    """Combine BCEWithLogits and Dice losses."""
    return bce_loss(logits, targets) + dice_loss(logits, targets)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net on the test set.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size for evaluation.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize dimension.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
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
    """Validate numeric arguments before evaluation."""
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if args.image_size <= 0:
        raise ValueError(f"--image-size must be positive, got {args.image_size}.")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be non-negative, got {args.num_workers}.")


def build_dataloader(
    dataset: HouseSegmentationDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """Create a DataLoader for test-time evaluation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    bce_loss: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model and return mean loss, IoU, and Dice."""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = combined_loss(logits, masks, bce_loss)

        running_loss += loss.item()
        running_iou += binary_iou(logits, masks)
        running_dice += binary_dice(logits, masks)

    num_batches = max(len(dataloader), 1)
    return {
        "test_loss": running_loss / num_batches,
        "test_iou": running_iou / num_batches,
        "test_dice": running_dice / num_batches,
    }


def main() -> None:
    """Load the trained model, evaluate it, and save test metrics."""
    args = parse_args()
    validate_args(args)

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = HouseSegmentationDataset(
        images_dir=args.test_images_dir,
        masks_dir=args.test_masks_dir,
        image_size=args.image_size,
        augment=False,
    )
    test_loader = build_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)

    bce_loss = nn.BCEWithLogitsLoss()
    metrics = evaluate(model=model, dataloader=test_loader, bce_loss=bce_loss, device=device)

    output_path = metrics_dir / "test_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Test Set Evaluation")
    print(f"  Loss: {metrics['test_loss']:.4f}")
    print(f"  IoU:  {metrics['test_iou']:.4f}")
    print(f"  Dice: {metrics['test_dice']:.4f}")
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
