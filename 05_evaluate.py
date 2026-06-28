"""
05_evaluate.py
---------------
Evaluates the trained student model and generates:
  - Accuracy report per student
  - Confusion matrix heatmap
  - Training loss/accuracy curves (from CSV logs)
  - Summary stats for report/patent documentation

Usage:
    python 05_evaluate.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving to file
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 160


def load_all():
    model       = tf.keras.models.load_model("models/student_model.keras")
    with open("models/student_labels.json") as f:
        label_names = json.load(f)
    return model, label_names


def evaluate_model(model, label_names, data_dir="dataset/students"):
    gen = ImageDataGenerator(rescale=1.0/255)
    ds  = gen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    preds     = model.predict(ds, verbose=1)
    y_pred    = np.argmax(preds, axis=1)
    y_true    = ds.classes
    idx_names = [k for k, v in sorted(ds.class_indices.items(), key=lambda x: x[1])]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n[RESULT] Overall Accuracy: {acc*100:.2f}%")
    print("\n[RESULT] Per-student report:")
    print(classification_report(y_true, y_pred, target_names=idx_names))

    os.makedirs("evaluation", exist_ok=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(10, len(idx_names)), max(8, len(idx_names)-2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=idx_names, yticklabels=idx_names, ax=ax)
    ax.set_title("Confusion Matrix — Student Face Recognition", fontsize=14, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("evaluation/confusion_matrix.png", dpi=150)
    plt.close()
    print("[INFO] ✓ Saved evaluation/confusion_matrix.png")

    return acc, y_true, y_pred, idx_names


def plot_training_curves():
    """Plot loss/accuracy from the CSV logs saved during training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — Student Fine-Tuning", fontsize=14)

    colors = {"phase1": ("#4C72B0", "#DD8452"), "phase2": ("#55A868", "#C44E52")}

    for phase, (c_train, c_val) in colors.items():
        log_file = f"models/{phase}_log.csv"
        if not os.path.exists(log_file):
            continue

        df = pd.read_csv(log_file)
        label = "Phase 1 (head)" if phase == "phase1" else "Phase 2 (fine-tune)"

        axes[0].plot(df["accuracy"],     color=c_train, label=f"{label} train", linewidth=2)
        axes[0].plot(df["val_accuracy"], color=c_val,   label=f"{label} val",   linewidth=2, linestyle="--")
        axes[1].plot(df["loss"],         color=c_train, label=f"{label} train", linewidth=2)
        axes[1].plot(df["val_loss"],     color=c_val,   label=f"{label} val",   linewidth=2, linestyle="--")

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation/training_curves.png", dpi=150)
    plt.close()
    print("[INFO] ✓ Saved evaluation/training_curves.png")


def generate_report(acc, label_names):
    """Generate a summary text report."""
    report = f"""
FACE RECOGNITION ATTENDANCE SYSTEM
Evaluation Report
{'='*50}

Model Architecture : Custom CNN (VGGFace2 pretrained → fine-tuned)
Number of Students : {len(label_names)}
Overall Accuracy   : {acc*100:.2f}%
Input Image Size   : 160 x 160 RGB

Students enrolled:
{chr(10).join(f'  - {n}' for n in label_names)}

Files:
  models/student_model.keras       <- trained model
  models/student_labels.json       <- student names
  models/student_encodings.pkl     <- face embeddings
  evaluation/confusion_matrix.png  <- confusion matrix
  evaluation/training_curves.png   <- training curves

Recognition Strategy:
  Dual check: CNN softmax (threshold 0.75) + 
              Embedding distance (threshold 0.55)
  Both must agree to mark attendance.
{'='*50}
"""
    with open("evaluation/report.txt", "w") as f:
        f.write(report)
    print(report)
    print("[INFO] ✓ Saved evaluation/report.txt")


if __name__ == "__main__":
    print("[INFO] Loading model...")
    model, label_names = load_all()

    print("[INFO] Evaluating on student dataset...")
    acc, y_true, y_pred, idx_names = evaluate_model(model, label_names)

    print("[INFO] Plotting training curves...")
    plot_training_curves()

    generate_report(acc, label_names)

    print("\n[INFO] All evaluation files saved to evaluation/")
