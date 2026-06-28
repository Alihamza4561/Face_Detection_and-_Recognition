"""
01_train_vggface2.py
---------------------
Stage 1: Train a CNN FROM SCRATCH on VGGFace2.

The CNN learns general face features:
  - Eye shapes, nose structure, jawlines, skin texture
  - Lighting variation, pose variation
  - What makes faces DIFFERENT from each other

This trained model becomes the BASE for student fine-tuning.

Architecture: Custom FaceNet-inspired CNN
  Input 160x160 RGB → Conv blocks → Embedding layer → Softmax (VGGFace2 classes)

After fine-tuning on students, we REMOVE the softmax head and use
the embedding layer directly for recognition (like FaceNet).

Usage:
    python 01_train_vggface2.py --data_dir dataset/vggface2_subset --epochs 30

Recommended hardware:
    GPU (NVIDIA): trains in 1–3 hours for 500 people × 50 images
    CPU only:     trains in 6–12 hours (use --people 100 for faster testing)

Output:
    models/vggface2_base.keras   <- base model with softmax head
    models/vggface2_encoder.keras <- embedding model (no head) for fine-tuning
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ── Constants ──────────────────────────────────────────────────────────────
IMG_SIZE    = 160   # FaceNet standard input size
BATCH_SIZE  = 64
EMBED_DIM   = 128   # size of face embedding vector


# ── Model Architecture ─────────────────────────────────────────────────────
def build_face_cnn(num_classes: int, embed_dim: int = EMBED_DIM):
    """
    Custom CNN architecture inspired by FaceNet/VGGFace.
    Built entirely from scratch — no pretrained weights.

    Structure:
      Block 1: 32 filters  — detects edges, basic textures
      Block 2: 64 filters  — detects facial parts (eyes, nose)
      Block 3: 128 filters — detects facial structures
      Block 4: 256 filters — detects high-level face identity features
      Embedding: 128-d vector — the face 'fingerprint'
      Head: Softmax over VGGFace2 classes (removed later for fine-tuning)
    """
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")

    # ── Block 1 ──────────────────────────────────────
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(0.1)(x)

    # ── Block 2 ──────────────────────────────────────
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Dropout(0.15)(x)

    # ── Block 3 ──────────────────────────────────────
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3_3")(x)
    x = layers.BatchNormalization(name="bn3_3")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)
    x = layers.Dropout(0.2)(x)

    # ── Block 4 ──────────────────────────────────────
    x = layers.Conv2D(256, 3, padding="same", use_bias=False, name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False, name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool4")(x)
    x = layers.Dropout(0.25)(x)

    # ── Embedding Layer ───────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, use_bias=False, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    # L2-normalized embedding — this is the face fingerprint
    embedding = layers.Dense(embed_dim, name="embedding")(x)
    embedding = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name="l2_norm"
    )(embedding)

    # Classification head (only used during VGGFace2 training)
    output = layers.Dense(num_classes, activation="softmax", name="classifier")(embedding)

    model = models.Model(inputs, output, name="FaceCNN")
    return model


def get_encoder(full_model):
    """Strip the classifier head → returns embedding model."""
    return models.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer("l2_norm").output,
        name="FaceEncoder"
    )


# ── Data Pipeline ──────────────────────────────────────────────────────────
def build_data_generators(data_dir: str, val_split: float = 0.15):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.15,
        brightness_range=[0.7, 1.3],
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
    )

    val_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
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


# ── Training ───────────────────────────────────────────────────────────────
def train(args):
    print(f"\n[INFO] Loading data from: {args.data_dir}")
    train_ds, val_ds = build_data_generators(args.data_dir)
    num_classes = len(train_ds.class_indices)
    print(f"[INFO] {num_classes} people | "
          f"{train_ds.samples} train images | {val_ds.samples} val images")

    print("\n[INFO] Building model...")
    model = build_face_cnn(num_classes=num_classes, embed_dim=EMBED_DIM)
    model.summary()

    # Cosine decay schedule — warms up then decays learning rate
    total_steps = (train_ds.samples // BATCH_SIZE) * args.epochs
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=1e-6,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )

    os.makedirs("models", exist_ok=True)

    cbs = [
        callbacks.ModelCheckpoint(
            "models/vggface2_base.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.CSVLogger("models/vggface2_training_log.csv"),
    ]

    print("\n[INFO] Training started...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cbs,
    )

    # Save encoder (no classification head) for fine-tuning on students
    encoder = get_encoder(model)
    encoder.save("models/vggface2_encoder.keras")

    print("\n[INFO] ✓ Saved: models/vggface2_base.keras    (full model)")
    print("[INFO] ✓ Saved: models/vggface2_encoder.keras  (encoder for fine-tuning)")
    print("[INFO] ✓ Saved: models/vggface2_training_log.csv")

    final_val_acc = max(history.history["val_accuracy"])
    print(f"\n[INFO] Best validation accuracy: {final_val_acc*100:.1f}%")
    print("\n[NEXT] Run: python 02_collect_students.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/vggface2_subset",
                        help="Path to prepared VGGFace2 subset")
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--lr",       type=float, default=1e-3,
                        help="Initial learning rate")
    args = parser.parse_args()

    # Use GPU if available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[INFO] GPU detected: {gpus}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("[WARN] No GPU detected — training on CPU (will be slow)")
        print("[WARN] Consider using Google Colab (free GPU) for Stage 1")

    train(args)
