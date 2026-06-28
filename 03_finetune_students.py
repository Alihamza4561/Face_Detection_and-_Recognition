"""
03_finetune_students.py
------------------------
Stage 2: Fine-tune the VGGFace2-trained CNN on your 80 students.

What happens:
  1. Load the encoder trained on VGGFace2 (already knows face features)
  2. FREEZE the Conv layers (keep VGGFace2 knowledge)
  3. Add a new classification head for YOUR students
  4. Train only the new head first (fast, avoids forgetting)
  5. UNFREEZE last Conv block and fine-tune everything together (slow but accurate)

This two-phase approach is critical:
  Phase 1 → new head learns student identities quickly
  Phase 2 → Conv layers adapt to your specific students/camera/lighting

Output:
  models/student_model.keras     <- full fine-tuned model
  models/student_labels.json     <- maps class index → student name
  models/student_encodings.pkl   <- precomputed embeddings for fast recognition

Usage:
    python 03_finetune_students.py --epochs_head 15 --epochs_full 20
"""

import os
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder


IMG_SIZE   = 160
BATCH_SIZE = 16   # small — student dataset is small
EMBED_DIM  = 128


def load_encoder():
    encoder_path = "models/vggface2_encoder.keras"
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Encoder not found at {encoder_path}\n"
            "Run 01_train_vggface2.py first!"
        )
    print(f"[INFO] Loading VGGFace2 encoder from {encoder_path}")
    return tf.keras.models.load_model(encoder_path)


def build_student_model(encoder, num_students: int):
    """
    Attach a new classification head on top of the VGGFace2 encoder.
    """
    # Freeze encoder initially
    encoder.trainable = False

    inputs    = encoder.input
    embedding = encoder.output   # 128-d L2 normalized vector

    # New head for students
    x = layers.Dense(256, activation="relu", name="student_dense1")(embedding)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", name="student_dense2")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_students, activation="softmax", name="student_out")(x)

    model = models.Model(inputs, output, name="StudentAttendanceModel")
    return model


def build_generators(data_dir: str):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.05,
        height_shift_range=0.05,
    )

    val_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_ds = train_gen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_ds = val_gen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    return train_ds, val_ds


def save_encodings(model, data_dir: str, label_names: list):
    """
    Precompute and save face embeddings for all student images.
    Used during recognition for fast comparison.
    """
    print("\n[INFO] Precomputing student face embeddings...")

    # Extract just the encoder part (up to l2_norm layer)
    encoder_only = models.Model(
        inputs=model.input,
        outputs=model.get_layer("l2_norm").output
    )

    all_encodings = []
    all_names     = []

    gen = ImageDataGenerator(rescale=1.0/255)

    for student_name in sorted(os.listdir(data_dir)):
        student_dir = os.path.join(data_dir, student_name)
        if not os.path.isdir(student_dir):
            continue

        images = [f for f in os.listdir(student_dir) if f.lower().endswith((".jpg", ".png"))]
        if not images:
            continue

        student_encodings = []
        for fname in images:
            img_path = os.path.join(student_dir, fname)
            img      = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(IMG_SIZE, IMG_SIZE)
            )
            img_arr  = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_arr  = np.expand_dims(img_arr, axis=0)

            encoding = encoder_only.predict(img_arr, verbose=0)[0]
            student_encodings.append(encoding)

        # Use mean encoding of all images as the student's representative embedding
        mean_encoding = np.mean(student_encodings, axis=0)
        # Re-normalize after averaging
        mean_encoding = mean_encoding / np.linalg.norm(mean_encoding)

        all_encodings.append(mean_encoding)
        all_names.append(student_name)
        print(f"  [✓] {student_name}: {len(student_encodings)} images encoded")

    os.makedirs("models", exist_ok=True)
    with open("models/student_encodings.pkl", "wb") as f:
        pickle.dump({"encodings": all_encodings, "names": all_names}, f)

    print(f"[INFO] ✓ Saved embeddings for {len(all_names)} students")


def train(args):
    train_ds, val_ds = build_generators(args.data_dir)
    num_students     = len(train_ds.class_indices)
    label_names      = [k for k, v in sorted(train_ds.class_indices.items(), key=lambda x: x[1])]

    print(f"[INFO] Fine-tuning for {num_students} students")
    print(f"[INFO] Students: {label_names}")
    print(f"[INFO] Train: {train_ds.samples} images | Val: {val_ds.samples} images")

    encoder = load_encoder()
    model   = build_student_model(encoder, num_students)

    os.makedirs("models", exist_ok=True)

    # ── Phase 1: Train head only (encoder frozen) ──────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1: Training new student head (encoder frozen)")
    print(f"{'='*60}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
            callbacks.CSVLogger("models/phase1_log.csv"),
        ],
    )

    # ── Phase 2: Unfreeze last conv block + fine-tune ─────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2: Fine-tuning (last conv block unfrozen)")
    print(f"{'='*60}")

    # Unfreeze last conv block (conv4) and everything after
    encoder.trainable = True
    for layer in model.layers:
        if hasattr(layer, "layers"):  # it's the encoder sub-model
            for enc_layer in layer.layers:
                enc_layer.trainable = enc_layer.name.startswith(
                    ("conv4", "bn4", "gap", "dense1", "bn_dense", "embedding", "l2_norm")
                )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # very low LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_full,
        callbacks=[
            callbacks.ModelCheckpoint(
                "models/student_model.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            callbacks.EarlyStopping(monitor="val_accuracy", patience=7,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
            callbacks.CSVLogger("models/phase2_log.csv"),
        ],
    )

    # Save label mapping
    with open("models/student_labels.json", "w") as f:
        json.dump(label_names, f, indent=2)
    print(f"\n[INFO] ✓ Saved student_labels.json")

    # Precompute and save embeddings for fast recognition
    save_encodings(model, args.data_dir, label_names)

    print("\n[INFO] ✓ Fine-tuning complete!")
    print("[INFO] Files saved:")
    print("       models/student_model.keras")
    print("       models/student_labels.json")
    print("       models/student_encodings.pkl")
    print("\n[NEXT] Run: python 04_attendance.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="dataset/students")
    parser.add_argument("--epochs_head", type=int, default=15,
                        help="Epochs for phase 1 (head only)")
    parser.add_argument("--epochs_full", type=int, default=20,
                        help="Epochs for phase 2 (full fine-tune)")
    args = parser.parse_args()
    train(args)
