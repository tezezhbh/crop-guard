"""
model.py — CropGuard AI
MobileNetV2 transfer learning — matches Colab notebook training exactly.
Mekelle Institute of Technology 2026

Architecture:
  - MobileNetV2 base: top 50 layers trainable, rest frozen
  - Custom head: GAP → BN → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.25) → Softmax
  - Single-phase training, 20 epochs, Adam(3e-4)

This matches the model that achieved 99.16% validation accuracy.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
IMG_SIZE       = (224, 224)
INPUT_SHAPE    = (*IMG_SIZE, 3)
LEARNING_RATE  = 3e-4       # matches notebook exactly
DROPOUT_RATE   = 0.4
DROPOUT_RATE2  = 0.25
DENSE_UNITS_1  = 512        # matches notebook
DENSE_UNITS_2  = 256
UNFREEZE_LAST  = 50         # unfreeze last 50 layers of MobileNetV2
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def build_mobilenetv2(num_classes: int) -> tf.keras.Model:
    """
    MobileNetV2 with partial fine-tuning from the start.
    Top 50 layers trainable, rest frozen. Matches Colab notebook.

    Args:
        num_classes: Derived from train_gen.num_classes — NEVER hardcode.
    """
    base = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )

    # Partial freeze — last 50 layers trainable, matches notebook
    for layer in base.layers[:-UNFREEZE_LAST]:
        layer.trainable = False
    for layer in base.layers[-UNFREEZE_LAST:]:
        layer.trainable = True

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"[model] MobileNetV2: {trainable_count}/{len(base.layers)} layers trainable (last {UNFREEZE_LAST})")

    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(DENSE_UNITS_1, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(DENSE_UNITS_2, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_CropGuard")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_callbacks(checkpoint_path: str):
    """Single checkpoint path — best model across all epochs."""
    return [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir="logs/mobilenetv2",
            histogram_freq=1,
        ),
    ]


def train(model, train_gen, val_gen, class_weights=None):
    """Single-phase training matching the notebook."""
    ckpt = str(CHECKPOINT_DIR / "mobilenetv2_best.keras")

    print("\n=== Training MobileNetV2 (single phase, matches Colab notebook) ===")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        class_weight=class_weights,
        callbacks=get_callbacks(ckpt),
        verbose=1,
    )

    best_acc = max(history.history["val_accuracy"])
    print(f"\n[model] Best val_accuracy: {best_acc*100:.2f}%")
    print(f"[model] Best checkpoint saved at: {ckpt}")
    return model, history


def export_model(model, class_indices, path="saved_model/plant_disease_model"):
    """
    Export to TensorFlow SavedModel format.
    Also saves class_index.json alongside the model variables
    so serve.py always finds the correct matching index.

    Args:
        class_indices: dict from train_gen.class_indices — no re-scanning, no desync risk.
    """
    import json
    Path(path).mkdir(parents=True, exist_ok=True)
    model.export(path)

    # Save class index using the training-time class mapping (no re-scanning)
    idx = {str(v): k for k, v in class_indices.items()}
    idx_path = Path(path) / "variables" / "class_index.json"
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)

    print(f"[model] Exported to: {path}")
    print(f"[model] Class index saved to: {idx_path}")
    print(f"[model] Classes: {len(idx)}")


if __name__ == "__main__":
    import json
    from preprocessing import (
        build_generators, compute_class_weights,
        TRAIN_DIR, VAL_DIR, TEST_DIR,
    )

    # Build generators
    train_gen, val_gen, test_gen = build_generators(TRAIN_DIR, VAL_DIR, TEST_DIR)

    # ALWAYS derive from data — never hardcode
    NUM_CLASSES = train_gen.num_classes
    print(f"[model] Training for {NUM_CLASSES} classes")
    print(f"[model] Train: {train_gen.samples} | Val: {val_gen.samples}")

    class_weights = compute_class_weights(train_gen)

    # Build and train
    model = build_mobilenetv2(NUM_CLASSES)
    model.summary()

    model, history = train(model, train_gen, val_gen, class_weights)

    # Export (includes saving class_index.json)
    export_model(model, train_gen.class_indices)
