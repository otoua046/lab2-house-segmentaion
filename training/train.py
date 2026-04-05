"""Train a lightweight U-Net for binary house segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    """A reasonably lightweight U-Net for binary segmentation."""

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
        """Resize decoder features when spatial sizes differ slightly."""
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
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.float().flatten(start_dim=1)

    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice_score.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor, bce_loss: nn.Module) -> torch.Tensor:
    """Combine BCEWithLogits and Dice losses."""
    return bce_loss(logits, targets) + dice_loss(logits, targets)


def build_dataloader(dataset: HouseSegmentationDataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    """Create a DataLoader with sensible CPU/GPU defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_loss: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return mean training loss."""
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = combined_loss(logits, masks, bce_loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    bce_loss: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one validation epoch and return loss, IoU, and Dice."""
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
    return (
        running_loss / num_batches,
        running_iou / num_batches,
        running_dice / num_batches,
    )


def save_training_curves(history: dict[str, list[float]], plots_dir: Path) -> None:
    """Save loss and Dice curves as PNG files."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_dice"], label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "dice_curve.png")
    plt.close()


def save_history(history: dict[str, list[float]], metrics_dir: Path, best_epoch: int, best_val_dice: float) -> None:
    """Save training metrics history to JSON."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **history,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
    }
    with (metrics_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train a binary segmentation U-Net.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--image-size", type=int, default=256, help="Square image size for resizing.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument(
        "--train-images-dir",
        type=Path,
        default=Path("data/train/images"),
        help="Path to training images.",
    )
    parser.add_argument(
        "--train-masks-dir",
        type=Path,
        default=Path("data/train/masks"),
        help="Path to training masks.",
    )
    parser.add_argument(
        "--val-images-dir",
        type=Path,
        default=Path("data/val/images"),
        help="Path to validation images.",
    )
    parser.add_argument(
        "--val-masks-dir",
        type=Path,
        default=Path("data/val/masks"),
        help="Path to validation masks.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/best_model.pth"),
        help="Where to save the best model checkpoint.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate numeric CLI inputs before training starts."""
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if args.lr <= 0:
        raise ValueError(f"--lr must be positive, got {args.lr}.")
    if args.image_size <= 0:
        raise ValueError(f"--image-size must be positive, got {args.image_size}.")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be non-negative, got {args.num_workers}.")


def main() -> None:
    """Train the model, evaluate each epoch, and save outputs."""
    args = parse_args()
    validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plots_dir = Path("outputs/plots")
    metrics_dir = Path("outputs/metrics")
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = HouseSegmentationDataset(
        images_dir=args.train_images_dir,
        masks_dir=args.train_masks_dir,
        image_size=args.image_size,
        augment=True,
    )
    val_dataset = HouseSegmentationDataset(
        images_dir=args.val_images_dir,
        masks_dir=args.val_masks_dir,
        image_size=args.image_size,
        augment=False,
    )

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce_loss = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
    }
    best_val_dice = float("-inf")
    best_epoch = 0

    print(
        f"Starting training on {device.type.upper()} | "
        f"train_samples={len(train_dataset)} | val_samples={len(val_dataset)} | "
        f"epochs={args.epochs} | batch_size={args.batch_size} | lr={args.lr}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            bce_loss=bce_loss,
            device=device,
        )
        val_loss, val_iou, val_dice = validate(
            model=model,
            dataloader=val_loader,
            bce_loss=bce_loss,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | "
            f"val_dice={val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), args.model_path)
            print(f"  Saved new best model to {args.model_path} (val_dice={val_dice:.4f})")

    save_training_curves(history, plots_dir)
    save_history(history, metrics_dir, best_epoch=best_epoch, best_val_dice=best_val_dice)

    print(f"Training complete. Best validation Dice: {best_val_dice:.4f} at epoch {best_epoch}.")
    print(f"Saved plots to {plots_dir} and history JSON to {metrics_dir / 'training_history.json'}.")


if __name__ == "__main__":
    main()
