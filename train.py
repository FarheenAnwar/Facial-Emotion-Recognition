"""
train.py — Train a CNN on FER2013 (folder layout) for 7-class emotion recognition.

Dataset layout expected:
    dataset/train/<emotion>/  ← training images per class
    dataset/test/<emotion>/   ← validation / test images per class

Outputs:
    models/emotion_model.h5        ← best checkpoint
    models/class_indices.json      ← {emotion: index} mapping
    models/training_history.png
    models/confusion_matrix.png

Usage:
    python train.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Dropout,
    Flatten, Dense, Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE           = 48
BATCH_SIZE         = 64
EPOCHS             = 50
TRAIN_DIR          = "dataset/train"
TEST_DIR           = "dataset/test"
MODEL_PATH         = "models/emotion_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"


# ── Data generators ────────────────────────────────────────────────────────────

def get_generators():
    """Return (train_gen, val_gen) using ImageDataGenerator.flow_from_directory."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        color_mode="grayscale",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        TEST_DIR,
        color_mode="grayscale",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,  # keep order for confusion-matrix computation
    )
    return train_gen, val_gen


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 7) -> Sequential:
    """
    Three convolutional blocks (64 → 128 → 256 filters), each followed by
    BatchNorm, MaxPool, and Dropout, then a two-layer dense head.
    """
    model = Sequential(
        [
            Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

            # ── Block 1 ────────────────────────────────────────────────────────
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # ── Block 2 ────────────────────────────────────────────────────────
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # ── Block 3 ────────────────────────────────────────────────────────
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # ── Dense head ─────────────────────────────────────────────────────
            Flatten(),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ],
        name="EmotionCNN",
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ── Callbacks ──────────────────────────────────────────────────────────────────

def get_callbacks():
    os.makedirs("models", exist_ok=True)
    return [
        ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1,
        ),
    ]


# ── Post-training diagnostics ──────────────────────────────────────────────────

def save_training_plots(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"],     label="Train Acc",  color="royalblue")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    color="darkorange")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss", color="royalblue")
    axes[1].plot(history.history["val_loss"], label="Val Loss",   color="darkorange")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("models/training_history.png", dpi=150)
    plt.close()
    print("Saved  models/training_history.png")


def save_confusion_matrix(model, val_gen, class_names: list):
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes[: len(y_pred)]    # guard against uneven last batch

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved  models/confusion_matrix.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    for path in [TRAIN_DIR, TEST_DIR]:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Directory not found: '{path}'\n"
                "Place FER2013 images under dataset/train/<emotion>/ and dataset/test/<emotion>/."
            )

    train_gen, val_gen = get_generators()

    # Persist label→index mapping so predict.py / webcam.py never re-derive it
    os.makedirs("models", exist_ok=True)
    with open(CLASS_INDICES_PATH, "w") as fh:
        json.dump(train_gen.class_indices, fh, indent=2)
    print(f"\nClass indices: {train_gen.class_indices}")
    print(f"Saved  {CLASS_INDICES_PATH}\n")

    # Sorted class name list (index order)
    class_names = [
        k for k, _ in sorted(train_gen.class_indices.items(), key=lambda x: x[1])
    ]

    model = build_model(num_classes=len(class_names))

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
        verbose=1,
    )

    print("\nFinal evaluation on test set …")
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"  Test accuracy : {acc * 100:.2f}%")
    print(f"  Test loss     : {loss:.4f}")

    save_training_plots(history)
    save_confusion_matrix(model, val_gen, class_names)
    print(f"\nDone. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
