"""
preprocessing.py — CropGuard AI
Mekelle Institute of Technology 2026

Fixes over v1:
  - interpolation='bilinear' on all generators (matches PIL.BILINEAR in serve.py)
  - Guard against re-splitting an existing dataset
  - Derives paths relative to this file so it works from any cwd

Shared preprocessing:
  - preprocess_image(): unified function for inference (matches training pipeline)
  - MobileNetV2 preprocess_input NOT used to preserve compatibility with trained weights
    (model was trained with [0,1] range, not [-1,1] that preprocess_input produces)
"""

import shutil
import random
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Configuration ──────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)   # MobileNetV2 input size
BATCH_SIZE = 64           # matches Colab notebook (was 32 locally)
SEED       = 42

_HERE     = Path(__file__).parent
DATA_DIR  = _HERE / "data" / "plantvillage"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "val"
TEST_DIR  = DATA_DIR / "test"

# ── Shared preprocessing (training + inference) ─────────────────────────
def preprocess_image(image_bytes: bytes):
    """
    Unified preprocessing for inference.
    Matches ImageDataGenerator pipeline used in training:
      - PIL BILINEAR resize to 224x224
      - Normalise to [0, 1] (rescale=1/255.0)

    NOTE: MobileNetV2 preprocess_input is NOT used here to preserve
    compatibility with the existing trained weights. The model was trained
    with [0,1] range, not the [-1,1] range that preprocess_input produces.
    """
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]

# ── Augmentation ───────────────────────────────────────────────────────────
# Matches Colab notebook augmentation exactly.
# interpolation='bilinear' ensures same pixel pipeline as serve.py
train_datagen = ImageDataGenerator(
    rescale            = 1.0 / 255.0,
    rotation_range     = 35,
    zoom_range         = 0.2,
    horizontal_flip    = True,
    brightness_range   = [0.75, 1.25],
    fill_mode          = "nearest",
    # Not in original: added for real-world farm photo robustness
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
)

eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)


def build_generators(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR):
    """Build train/val/test generators with consistent bilinear interpolation."""
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size   = IMG_SIZE,
        batch_size    = BATCH_SIZE,
        class_mode    = "categorical",
        shuffle       = True,
        seed          = SEED,
        interpolation = "bilinear",  # matches PIL.BILINEAR in serve.py
    )
    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size   = IMG_SIZE,
        batch_size    = BATCH_SIZE,
        class_mode    = "categorical",
        shuffle       = False,
        interpolation = "bilinear",
    )
    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size   = IMG_SIZE,
        batch_size    = BATCH_SIZE,
        class_mode    = "categorical",
        shuffle       = False,
        interpolation = "bilinear",
    )
    return train_gen, val_gen, test_gen


def split_dataset(source_dir: Path, split=(0.70, 0.15, 0.15)):
    """
    Splits a flat directory into train/val/test subfolders.

    source_dir/
        class_a/  img1.jpg  img2.jpg ...
        class_b/  ...
    """
    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    assert not TRAIN_DIR.exists(), (
        f"Train dir already exists at {TRAIN_DIR}. "
        "Delete data/plantvillage/ completely before re-splitting."
    )

    for class_folder in sorted(source_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        images = sorted(
            list(class_folder.glob("*.jpg")) +
            list(class_folder.glob("*.JPG")) +
            list(class_folder.glob("*.png"))
        )
        random.seed(SEED)
        random.shuffle(images)

        n    = len(images)
        n_tr = int(n * split[0])
        n_va = int(n * split[1])

        for subset, files in [
            ("train", images[:n_tr]),
            ("val",   images[n_tr:n_tr + n_va]),
            ("test",  images[n_tr + n_va:]),
        ]:
            dest = DATA_DIR / subset / class_folder.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)

    print("[✓] Dataset split complete")


def compute_class_weights(train_gen):
    """Balanced class weights to handle imbalanced disease classes."""
    from sklearn.utils.class_weight import compute_class_weight
    labels  = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


if __name__ == "__main__":
    if not TRAIN_DIR.exists():
        split_dataset(DATA_DIR)

    train_gen, val_gen, test_gen = build_generators()

    print(f"Classes  : {train_gen.num_classes}")
    print(f"Train    : {train_gen.samples} images")
    print(f"Val      : {val_gen.samples} images")
    print(f"Test     : {test_gen.samples} images")

    imgs, labels = next(train_gen)
    print(f"Batch    : images {imgs.shape}, labels {labels.shape}")
    print(f"Pixel range: [{imgs.min():.3f}, {imgs.max():.3f}]")
    assert imgs.max() <= 1.0, "Pixel values not normalised!"
    print("[✓] Sanity check passed")
