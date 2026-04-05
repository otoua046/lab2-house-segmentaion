"""Split paired raw images and masks into train/val/test directories."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PairedSample:
    """Represents one image file and its matching mask file."""

    stem: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split paired raw images and masks into train/val/test sets."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw images/ and masks/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where train/, val/, and test/ folders will be created.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of samples assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of samples assigned to the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of samples assigned to the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic shuffling.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="Whether to copy or move files into the split directories.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate split ratios before any file operations begin."""
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    for split_name, ratio in ratios.items():
        if ratio < 0:
            raise ValueError(f"{split_name} ratio must be non-negative, got {ratio}.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Received train={train_ratio}, val={val_ratio}, test={test_ratio} "
            f"(sum={ratio_sum})."
        )


def collect_files_by_stem(directory: Path) -> dict[str, Path]:
    """Map file stems to paths and reject duplicate stems within the same folder."""
    if not directory.exists():
        raise FileNotFoundError(f"Required directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {directory}")

    files_by_stem: dict[str, Path] = {}
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        if path.stem in files_by_stem:
            existing = files_by_stem[path.stem]
            raise ValueError(
                f"Duplicate stem '{path.stem}' found in {directory}: "
                f"{existing.name} and {path.name}"
            )
        files_by_stem[path.stem] = path
    return files_by_stem


def build_paired_samples(raw_dir: Path) -> list[PairedSample]:
    """Verify strict one-to-one image/mask pairing by filename stem."""
    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "masks"

    images_by_stem = collect_files_by_stem(images_dir)
    masks_by_stem = collect_files_by_stem(masks_dir)

    image_stems = set(images_by_stem)
    mask_stems = set(masks_by_stem)

    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)

    if missing_masks or missing_images:
        problems: list[str] = []
        if missing_masks:
            preview = ", ".join(missing_masks[:10])
            if len(missing_masks) > 10:
                preview += ", ..."
            problems.append(f"Missing masks for image stems: {preview}")
        if missing_images:
            preview = ", ".join(missing_images[:10])
            if len(missing_images) > 10:
                preview += ", ..."
            problems.append(f"Missing images for mask stems: {preview}")
        raise ValueError("Image/mask pairing mismatch. " + " | ".join(problems))

    paired_samples = [
        PairedSample(
            stem=stem,
            image_path=images_by_stem[stem],
            mask_path=masks_by_stem[stem],
        )
        for stem in sorted(image_stems)
    ]

    if not paired_samples:
        raise ValueError(f"No paired samples found in {raw_dir}.")

    return paired_samples


def compute_split_counts(
    total_samples: int, train_ratio: float, val_ratio: float, test_ratio: float
) -> dict[str, int]:
    """Convert split ratios into exact sample counts that sum to the dataset size."""
    del test_ratio

    train_count = int(total_samples * train_ratio)
    val_count = int(total_samples * val_ratio)
    test_count = total_samples - train_count - val_count

    return {
        "train": train_count,
        "val": val_count,
        "test": test_count,
    }


def assign_splits(
    paired_samples: list[PairedSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[PairedSample]]:
    """Shuffle paired samples deterministically and partition them by split."""
    shuffled_samples = paired_samples[:]
    random.Random(seed).shuffle(shuffled_samples)

    counts = compute_split_counts(
        total_samples=len(shuffled_samples),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_end = counts["train"]
    val_end = train_end + counts["val"]

    return {
        "train": shuffled_samples[:train_end],
        "val": shuffled_samples[train_end:val_end],
        "test": shuffled_samples[val_end:],
    }


def ensure_output_dirs(output_dir: Path) -> dict[str, dict[str, Path]]:
    """Create all split image/mask directories and return their paths."""
    split_dirs: dict[str, dict[str, Path]] = {}
    for split_name in SPLITS:
        images_dir = output_dir / split_name / "images"
        masks_dir = output_dir / split_name / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split_name] = {
            "images": images_dir,
            "masks": masks_dir,
        }
    return split_dirs


def transfer_file(source: Path, destination: Path, mode: str) -> None:
    """Copy or move a file to its target path."""
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "move":
        shutil.move(str(source), str(destination))
        return
    raise ValueError(f"Unsupported transfer mode: {mode}")


def transfer_split_samples(
    split_name: str,
    samples: list[PairedSample],
    split_dirs: dict[str, dict[str, Path]],
    mode: str,
) -> None:
    """Transfer image/mask pairs into one split while preserving filenames."""
    for sample in samples:
        image_destination = split_dirs[split_name]["images"] / sample.image_path.name
        mask_destination = split_dirs[split_name]["masks"] / sample.mask_path.name

        transfer_file(sample.image_path, image_destination, mode)
        transfer_file(sample.mask_path, mask_destination, mode)


def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    mode: str,
) -> dict[str, int]:
    """Validate paired files, split them deterministically, and transfer them."""
    validate_ratios(train_ratio, val_ratio, test_ratio)
    paired_samples = build_paired_samples(raw_dir)
    split_assignment = assign_splits(
        paired_samples=paired_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_dirs = ensure_output_dirs(output_dir)

    for split_name in SPLITS:
        transfer_split_samples(
            split_name=split_name,
            samples=split_assignment[split_name],
            split_dirs=split_dirs,
            mode=mode,
        )

    return {split_name: len(split_assignment[split_name]) for split_name in SPLITS}


def print_summary(summary: dict[str, int]) -> None:
    """Print a compact summary of sample counts per split."""
    total = sum(summary.values())
    print("Dataset split summary:")
    print(f"  train: {summary['train']}")
    print(f"  val:   {summary['val']}")
    print(f"  test:  {summary['test']}")
    print(f"  total: {total}")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    summary = prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        mode=args.mode,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
