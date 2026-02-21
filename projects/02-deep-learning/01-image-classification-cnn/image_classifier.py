"""
Image Classification with Convolutional Neural Networks (CNN)
=============================================================

A production-quality CNN pipeline for CIFAR-10 image classification.

Architecture:
    - Multiple Conv2D blocks with BatchNormalization and MaxPooling
    - Dropout regularization to prevent overfitting
    - Data augmentation (rotation, flip, zoom, shift)
    - Learning rate scheduling with ReduceLROnPlateau
    - Comprehensive evaluation with confusion matrix and per-class metrics

Dataset:
    CIFAR-10 - 60,000 32x32 color images in 10 classes

Author: Data Science Portfolio
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    """Central configuration for the image classification pipeline."""

    # Data
    NUM_CLASSES = 10
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    IMG_CHANNELS = 3

    # Training
    BATCH_SIZE = 64
    EPOCHS = 50
    INITIAL_LR = 1e-3
    VALIDATION_SPLIT = 0.1

    # Augmentation
    ROTATION_RANGE = 15
    WIDTH_SHIFT = 0.1
    HEIGHT_SHIFT = 0.1
    HORIZONTAL_FLIP = True
    ZOOM_RANGE = 0.1

    # Paths
    OUTPUT_DIR = Path("outputs")
    MODEL_PATH = OUTPUT_DIR / "cifar10_cnn_model.keras"
    HISTORY_PLOT = OUTPUT_DIR / "training_history.png"
    CONFUSION_MATRIX_PLOT = OUTPUT_DIR / "confusion_matrix.png"
    CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.txt"

    # CIFAR-10 class names
    CLASS_NAMES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------
def load_and_preprocess_data():
    """Load CIFAR-10 dataset and apply preprocessing.

    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
        Images are normalized to [0, 1]. Labels are one-hot encoded.
    """
    print("=" * 60)
    print("Loading CIFAR-10 Dataset")
    print("=" * 60)

    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Split training into train + validation
    val_size = int(len(x_train_full) * Config.VALIDATION_SPLIT)
    x_val = x_train_full[:val_size]
    y_val = y_train_full[:val_size]
    x_train = x_train_full[val_size:]
    y_train = y_train_full[val_size:]

    # One-hot encode labels
    y_train_cat = keras.utils.to_categorical(y_train, Config.NUM_CLASSES)
    y_val_cat = keras.utils.to_categorical(y_val, Config.NUM_CLASSES)
    y_test_cat = keras.utils.to_categorical(y_test, Config.NUM_CLASSES)

    print(f"  Training set   : {x_train.shape[0]:,} images")
    print(f"  Validation set : {x_val.shape[0]:,} images")
    print(f"  Test set       : {x_test.shape[0]:,} images")
    print(f"  Image shape    : {x_train.shape[1:]}")
    print(f"  Classes        : {Config.NUM_CLASSES}")
    print()

    return x_train, y_train_cat, x_val, y_val_cat, x_test, y_test_cat, y_test


def create_data_augmentation_generator():
    """Create an ImageDataGenerator with augmentation for training.

    Augmentations include rotation, shifting, flipping, and zooming
    to improve model generalization.

    Returns:
        ImageDataGenerator instance configured with augmentation.
    """
    datagen = ImageDataGenerator(
        rotation_range=Config.ROTATION_RANGE,
        width_shift_range=Config.WIDTH_SHIFT,
        height_shift_range=Config.HEIGHT_SHIFT,
        horizontal_flip=Config.HORIZONTAL_FLIP,
        zoom_range=Config.ZOOM_RANGE,
        fill_mode="nearest",
    )
    return datagen


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
def build_cnn_model():
    """Build a CNN model for CIFAR-10 classification.

    Architecture:
        Block 1: Conv2D(32) -> BN -> Conv2D(32) -> BN -> MaxPool -> Dropout(0.2)
        Block 2: Conv2D(64) -> BN -> Conv2D(64) -> BN -> MaxPool -> Dropout(0.3)
        Block 3: Conv2D(128) -> BN -> Conv2D(128) -> BN -> MaxPool -> Dropout(0.4)
        Head:    Flatten -> Dense(256) -> BN -> Dropout(0.5) -> Dense(10, softmax)

    Returns:
        Compiled Keras model.
    """
    model = models.Sequential(name="CIFAR10_CNN")

    # --- Block 1 ---
    model.add(layers.Conv2D(
        32, (3, 3), padding="same", activation="relu",
        input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS),
        name="conv1a",
    ))
    model.add(layers.BatchNormalization(name="bn1a"))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1b"))
    model.add(layers.BatchNormalization(name="bn1b"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))
    model.add(layers.Dropout(0.2, name="drop1"))

    # --- Block 2 ---
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2a"))
    model.add(layers.BatchNormalization(name="bn2a"))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2b"))
    model.add(layers.BatchNormalization(name="bn2b"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))
    model.add(layers.Dropout(0.3, name="drop2"))

    # --- Block 3 ---
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3a"))
    model.add(layers.BatchNormalization(name="bn3a"))
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3b"))
    model.add(layers.BatchNormalization(name="bn3b"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3"))
    model.add(layers.Dropout(0.4, name="drop3"))

    # --- Classification Head ---
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(256, activation="relu", name="dense1"))
    model.add(layers.BatchNormalization(name="bn_dense"))
    model.add(layers.Dropout(0.5, name="drop_dense"))
    model.add(layers.Dense(Config.NUM_CLASSES, activation="softmax", name="output"))

    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=Config.INITIAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def print_model_summary(model):
    """Print a formatted model architecture summary."""
    print("=" * 60)
    print("Model Architecture")
    print("=" * 60)
    model.summary()
    print()

    total_params = model.count_params()
    print(f"  Total parameters : {total_params:,}")
    print()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def get_callbacks():
    """Create training callbacks.

    Returns:
        List of Keras callbacks:
        - ReduceLROnPlateau: halve LR when val_loss stalls for 5 epochs
        - EarlyStopping: stop if val_loss doesn't improve for 10 epochs
        - ModelCheckpoint: save best model by val_accuracy
    """
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cb_list = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(Config.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]
    return cb_list


def train_model(model, x_train, y_train, x_val, y_val):
    """Train the CNN model with data augmentation.

    Args:
        model: Compiled Keras model.
        x_train: Training images, shape (N, 32, 32, 3).
        y_train: One-hot training labels, shape (N, 10).
        x_val: Validation images.
        y_val: One-hot validation labels.

    Returns:
        Keras History object containing training metrics.
    """
    print("=" * 60)
    print("Training Model")
    print("=" * 60)

    datagen = create_data_augmentation_generator()
    datagen.fit(x_train)

    train_flow = datagen.flow(x_train, y_train, batch_size=Config.BATCH_SIZE)
    steps_per_epoch = len(x_train) // Config.BATCH_SIZE

    history = model.fit(
        train_flow,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=get_callbacks(),
        verbose=1,
    )

    print(f"\n  Training complete. Best model saved to: {Config.MODEL_PATH}")
    print()
    return history


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_training_history(history):
    """Plot training and validation loss/accuracy curves.

    Creates a two-panel figure saved to Config.HISTORY_PLOT.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    axes[1].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.HISTORY_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training history plot saved to: {Config.HISTORY_PLOT}")


def plot_confusion_matrix(y_true, y_pred):
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: True class indices, shape (N,).
        y_pred: Predicted class indices, shape (N,).
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(Config.CONFUSION_MATRIX_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix plot saved to: {Config.CONFUSION_MATRIX_PLOT}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test_cat, y_test_raw):
    """Evaluate the model on the test set and produce detailed metrics.

    Args:
        model: Trained Keras model.
        x_test: Test images, shape (N, 32, 32, 3).
        y_test_cat: One-hot test labels.
        y_test_raw: Integer test labels, shape (N, 1).

    Returns:
        Tuple of (y_true, y_pred) integer arrays.
    """
    print("=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    # Overall metrics
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print()

    # Per-class metrics
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test_raw.flatten()

    report = classification_report(
        y_true, y_pred,
        target_names=Config.CLASS_NAMES,
        digits=4,
    )
    print("  Per-Class Classification Report:")
    print("-" * 60)
    print(report)

    # Save report to file
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(Config.CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write("CIFAR-10 CNN Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Accuracy : {test_acc:.4f}\n\n")
        f.write(report)
    print(f"  Report saved to: {Config.CLASSIFICATION_REPORT_PATH}")

    # Per-class accuracy breakdown
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None,
    )
    print("\n  Per-Class Accuracy Breakdown:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 54)
    for i, name in enumerate(Config.CLASS_NAMES):
        print(f"  {name:<12} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")
    print()

    return y_true, y_pred


def generate_sample_predictions(model, x_test, y_test_raw, num_samples=10):
    """Generate and display predictions on random test samples.

    Args:
        model: Trained Keras model.
        x_test: Test images.
        y_test_raw: Integer test labels.
        num_samples: Number of samples to display.
    """
    print("=" * 60)
    print(f"Sample Predictions ({num_samples} random test images)")
    print("=" * 60)

    indices = np.random.choice(len(x_test), num_samples, replace=False)
    predictions = model.predict(x_test[indices], verbose=0)

    for i, idx in enumerate(indices):
        true_label = Config.CLASS_NAMES[y_test_raw[idx].item()]
        pred_label = Config.CLASS_NAMES[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100
        status = "CORRECT" if true_label == pred_label else "WRONG"

        print(f"  [{i+1:>2}] True: {true_label:<12} | Predicted: {pred_label:<12} "
              f"| Confidence: {confidence:5.1f}% | {status}")
    print()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    """Execute the full image classification pipeline."""
    print("\n" + "=" * 60)
    print("  CIFAR-10 Image Classification with CNN")
    print("=" * 60 + "\n")

    # Reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create output directory
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    x_train, y_train, x_val, y_val, x_test, y_test_cat, y_test_raw = (
        load_and_preprocess_data()
    )

    # 2. Build model
    model = build_cnn_model()
    print_model_summary(model)

    # 3. Train
    history = train_model(model, x_train, y_train, x_val, y_val)

    # 4. Visualize training
    plot_training_history(history)

    # 5. Evaluate
    y_true, y_pred = evaluate_model(model, x_test, y_test_cat, y_test_raw)

    # 6. Confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # 7. Sample predictions
    generate_sample_predictions(model, x_test, y_test_raw)

    # 8. Save final model
    model.save(Config.MODEL_PATH)
    print(f"  Final model saved to: {Config.MODEL_PATH}")

    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
