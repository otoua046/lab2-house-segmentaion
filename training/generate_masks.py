"""Generate binary building masks from polygon annotations."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datasets import DatasetDict, load_dataset
from PIL import Image, ImageDraw

DATASET_NAME = "keremberke/satellite-building-segmentation"
DATASET_CONFIG = "full"
EXPECTED_SPLITS = ("train", "valid", "test")
SPLIT_ALIASES = {"valid": ("valid", "validation")}
FOREGROUND_VALUE = 255
BACKGROUND_VALUE = 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate binary segmentation masks from polygon annotations."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples to process across all splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where images/ and masks/ will be created.",
    )
    parser.add_argument(
        "--save-split-json",
        type=Path,
        default=None,
        help="Optional path to save the original split mapping for each image_id.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Create image and mask output directories if they do not exist."""
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, masks_dir


def is_number(value: Any) -> bool:
    """Return True when the value can be interpreted as a numeric coordinate."""
    return isinstance(value, (int, float, np.integer, np.floating))


def extract_polygons(segmentation: Any) -> list[list[float]]:
    """Normalize nested segmentation data into a list of flat polygon coordinate lists."""
    if segmentation is None:
        return []

    if isinstance(segmentation, np.ndarray):
        segmentation = segmentation.tolist()

    if not isinstance(segmentation, (list, tuple)):
        return []

    if segmentation and all(is_number(point) for point in segmentation):
        if len(segmentation) >= 6 and len(segmentation) % 2 == 0:
            return [[float(point) for point in segmentation]]
        return []

    polygons: list[list[float]] = []
    for item in segmentation:
        # Some datasets store one polygon per object, while others nest
        # multiple polygon rings inside deeper list structures.
        polygons.extend(extract_polygons(item))
    return polygons


def segmentation_to_mask(
    image_size: tuple[int, int], segmentations: Iterable[Any]
) -> Image.Image:
    """Rasterize all building polygons into a single binary mask."""
    width, height = image_size
    mask = Image.new("L", (width, height), color=BACKGROUND_VALUE)
    drawer = ImageDraw.Draw(mask)

    for segmentation in segmentations:
        for polygon in extract_polygons(segmentation):
            points = [
                (float(polygon[index]), float(polygon[index + 1]))
                for index in range(0, len(polygon), 2)
            ]
            if len(points) >= 3:
                # Fill every building polygon into the same semantic mask so the
                # final result is binary foreground vs. background.
                drawer.polygon(points, fill=FOREGROUND_VALUE, outline=FOREGROUND_VALUE)

    return mask


def save_image_and_mask(
    image: Image.Image, mask: Image.Image, images_dir: Path, masks_dir: Path, stem: str
) -> tuple[Path, Path]:
    """Save the source image and its generated binary mask using matching filenames."""
    image_path = images_dir / f"{stem}.png"
    mask_path = masks_dir / f"{stem}.png"

    image.convert("RGB").save(image_path, format="PNG")
    mask.convert("L").save(mask_path, format="PNG")

    return image_path, mask_path


def resolve_split_name(dataset: DatasetDict, expected_split: str) -> str | None:
    """Resolve the expected split to the actual split name present in the dataset."""
    candidates = SPLIT_ALIASES.get(expected_split, (expected_split,))
    for candidate in candidates:
        if candidate in dataset:
            return candidate
    return None


def build_image_id(example: dict[str, Any], split_name: str, index: int) -> str:
    """Extract a stable image identifier from the example, with a fallback when absent."""
    for key in ("image_id", "id"):
        value = example.get(key)
        if value is not None:
            return str(value)
    return f"{split_name}_{index}"


def generate_masks(
    limit: int | None = None,
    output_dir: Path = Path("data/raw"),
    save_split_json: Path | None = None,
) -> None:
    """Load all dataset splits, rasterize polygons, and save paired PNG files."""
    images_dir, masks_dir = ensure_output_dirs(output_dir)

    logging.info("Loading dataset: %s", DATASET_NAME)
    logging.info("Loading dataset: %s (config=%s)", DATASET_NAME, DATASET_CONFIG)
    dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG)

    total_available = 0
    split_order: list[tuple[str, str]] = []
    for expected_split in EXPECTED_SPLITS:
        actual_split = resolve_split_name(dataset, expected_split)
        if actual_split is None:
            logging.warning("Split '%s' was not found in the dataset and will be skipped.", expected_split)
            continue
        split_order.append((expected_split, actual_split))
        total_available += len(dataset[actual_split])

    if not split_order:
        raise RuntimeError("No supported dataset splits were found.")

    target_total = min(total_available, limit) if limit is not None else total_available
    filename_width = max(6, len(str(max(target_total, 1))))

    processed_count = 0
    skipped_count = 0
    split_mapping: dict[str, str] = {}

    for expected_split, actual_split in split_order:
        split_dataset = dataset[actual_split]
        logging.info(
            "Processing split '%s' (%d examples).",
            expected_split,
            len(split_dataset),
        )

        for index, example in enumerate(split_dataset):
            if limit is not None and processed_count >= limit:
                break

            try:
                image = example["image"]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(np.asarray(image))

                objects = example.get("objects") or {}
                segmentations = objects.get("segmentation") or []
                mask = segmentation_to_mask(image.size, segmentations)

                # Use a single global counter so image and mask filenames stay
                # aligned even when the data comes from different source splits.
                stem = f"{processed_count + 1:0{filename_width}d}"
                save_image_and_mask(image, mask, images_dir, masks_dir, stem)

                image_id = build_image_id(example, expected_split, index)
                split_mapping[image_id] = expected_split

                processed_count += 1
                if processed_count % 50 == 0:
                    logging.info("Saved %d/%d examples.", processed_count, target_total)
            except Exception as exc:  # noqa: BLE001
                skipped_count += 1
                image_id = build_image_id(example, expected_split, index)
                logging.warning(
                    "Skipping invalid example '%s' from split '%s': %s",
                    image_id,
                    expected_split,
                    exc,
                )
                continue

        if limit is not None and processed_count >= limit:
            break

    if save_split_json is not None:
        save_split_json.parent.mkdir(parents=True, exist_ok=True)
        with save_split_json.open("w", encoding="utf-8") as handle:
            json.dump(split_mapping, handle, indent=2, sort_keys=True)
        logging.info("Saved split mapping to %s", save_split_json)

    logging.info(
        "Finished generating masks. Saved %d examples and skipped %d invalid examples.",
        processed_count,
        skipped_count,
    )


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    generate_masks(
        limit=args.limit,
        output_dir=args.output_dir,
        save_split_json=args.save_split_json,
    )


if __name__ == "__main__":
    main()
