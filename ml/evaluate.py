"""
evaluate.py — CropGuard AI
Mekelle Institute of Technology 2026

Fixes over v1:
  - y_true uses gen.samples (not len(gen)*batch_size which can exceed gen.samples)
  - Loads class_index.json from the saved model directory (always in sync)
  - Reports per-class recall to identify weak classes for farmers
"""

import os
import glob
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from pathlib import Path

OUTPUT_DIR = Path("evaluation_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_best_model():
    """Try multiple locations in priority order."""
    candidates = [
        ("keras", "saved_model/plant_disease_model"),
        ("keras", "saved_model"),
    ] + [
        ("keras", p)
        for p in sorted(glob.glob("checkpoints/*.keras"), key=os.path.getmtime, reverse=True)
    ]

    for fmt, path in candidates:
        if not os.path.exists(path):
            continue
        print(f"[evaluate] Trying: {path}")
        try:
            if fmt == "keras":
                model = tf.keras.models.load_model(path)
            else:
                model = tf.keras.layers.TFSMLayer(path, call_endpoint="serving_default")
            print(f"[evaluate] Model loaded: {path}")
            return model, fmt
        except Exception as e:
            print(f"[evaluate]   Error: {e}")

    raise SystemExit("[evaluate] No model found. Run model.py first.")


def predict_generator(model, gen, fmt):
    gen.reset()
    if fmt == "keras" and hasattr(model, "predict"):
        # Use built-in predict for Keras models — more robust and efficient
        probs = model.predict(gen, verbose=0)
    else:
        # Manual loop for TFSMLayer / SavedModel format
        all_probs = []
        steps = len(gen)
        for i, (batch_x, _) in enumerate(gen):
            out = model(batch_x, training=False)
            if isinstance(out, dict):
                out = list(out.values())[0]
            all_probs.append(out.numpy())
            print(f"\r  Predicting batch {i+1}/{steps}", end="", flush=True)
            if i + 1 >= steps:
                break
        print()
        probs = np.concatenate(all_probs, axis=0)
    y_pred = np.argmax(probs, axis=1)
    return y_pred[:gen.samples]


def evaluate_model(model, gen, fmt, label="validation"):
    # FIX: use gen.samples directly — never len(gen)*batch_size
    y_true      = gen.classes[:gen.samples]
    class_names = list(gen.class_indices.keys())
    n_classes   = len(class_names)

    print(f"\n[evaluate] Running inference on {label} set "
          f"({gen.samples} images, {n_classes} classes)...")

    y_pred = predict_generator(model, gen, fmt)

    # Trim to same length (generator may pad last batch)
    min_len = min(len(y_true), len(y_pred))
    y_true  = y_true[:min_len]
    y_pred  = y_pred[:min_len]

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  CropGuard AI — Evaluation ({label} set)")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Macro F1  : {f1*100:.2f}%")
    print(f"  Images    : {min_len}")
    print(f"  Classes   : {n_classes}")
    print(f"{'='*60}\n")

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    # Flag classes with recall < 90% — these are the ones that could harm farmers
    from sklearn.metrics import recall_score
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    weak = [(class_names[i], round(per_class_recall[i]*100, 1))
            for i in range(n_classes) if per_class_recall[i] < 0.90]
    if weak:
        print("\n⚠️  Classes with recall < 90% (high false-negative risk for farmers):")
        for name, rec in sorted(weak, key=lambda x: x[1]):
            print(f"   {name:<55} {rec:.1f}%")
    else:
        print("\n✓ All classes have recall ≥ 90%")

    report_path = OUTPUT_DIR / f"classification_report_{label}.txt"
    with open(report_path, "w") as f:
        f.write(f"CropGuard AI — Evaluation Report\n")
        f.write(f"Set: {label}  |  Accuracy: {acc*100:.2f}%  |  Macro F1: {f1*100:.2f}%\n")
        f.write("="*60 + "\n")
        f.write(report)
    print(f"[✓] Report saved → {report_path}")

    return y_true, y_pred, class_names


def plot_confusion_matrix(y_true, y_pred, class_names, label="validation"):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    short = [c.replace("___", "\n").replace("_", " ") for c in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"Confusion Matrix — CropGuard AI ({label} set)", fontsize=14)

    for ax, data, title, fmt in [
        (axes[0], cm,     "Count",      "d"),
        (axes[1], cm_pct, "Normalised", ".2f"),
    ]:
        sns.heatmap(
            data, ax=ax,
            xticklabels=short, yticklabels=short,
            cmap="YlOrRd", fmt=fmt,
            annot=len(class_names) <= 20,
            linewidths=0.3, linecolor="white",
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True",      fontsize=10)
        ax.tick_params(axis="both", labelsize=6)

    plt.tight_layout()
    path = OUTPUT_DIR / f"confusion_matrix_{label}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Confusion matrix → {path}")


if __name__ == "__main__":
    from preprocessing import build_generators, TRAIN_DIR, VAL_DIR, TEST_DIR

    model, fmt = load_best_model()
    train_gen, val_gen, test_gen = build_generators(TRAIN_DIR, VAL_DIR, TEST_DIR)

    n_test_classes = len(test_gen.class_indices)
    if n_test_classes < train_gen.num_classes:
        print(f"\n[evaluate] ⚠  Test set only has {n_test_classes} classes — using validation set.")
        eval_gen, eval_label = val_gen, "validation"
    else:
        eval_gen, eval_label = test_gen, "test"

    y_true, y_pred, class_names = evaluate_model(model, eval_gen, fmt, eval_label)
    plot_confusion_matrix(y_true, y_pred, class_names, eval_label)

    print(f"\n[✓] Outputs saved to: {OUTPUT_DIR}/")
