"""
Face Mask Detection with Transfer Learning
===========================================

A production-quality pipeline for binary face mask classification
using a MobileNetV2-style transfer learning architecture.

Architecture:
    - MobileNetV2 backbone (frozen) pretrained on ImageNet
    - Custom classification head with GlobalAveragePooling, Dense, Dropout
    - Fine-tuning of top backbone layers after initial training
    - Comprehensive data augmentation pipeline

Dataset:
    Synthetic image features simulating face images with/without masks.
    Features are generated to mimic MobileNetV2 intermediate representations.

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    """Central configuration for the face mask detection pipeline."""

    # Data
    NUM_SAMPLES = 8000
    IMG_SIZE = 96
    IMG_CHANNELS = 3

    # Training
    BATCH_SIZE = 32
    EPOCHS_PHASE1 = 20  # Train head only
    EPOCHS_PHASE2 = 15  # Fine-tune top layers
    INITIAL_LR = 1e-3
    FINE_TUNE_LR = 1e-5
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # Augmentation
    ROTATION_RANGE = 20
    WIDTH_SHIFT = 0.15
    HEIGHT_SHIFT = 0.15
    HORIZONTAL_FLIP = True
    ZOOM_RANGE = 0.15
    BRIGHTNESS_RANGE = (0.8, 1.2)

    # Paths
    OUTPUT_DIR = Path("outputs")
    MODEL_PATH = OUTPUT_DIR / "mask_detector_model.keras"
    HISTORY_PLOT = OUTPUT_DIR / "training_history.png"
    CONFUSION_MATRIX_PLOT = OUTPUT_DIR / "confusion_matrix.png"
    ROC_PLOT = OUTPUT_DIR / "roc_curve.png"
    PR_PLOT = OUTPUT_DIR / "precision_recall_curve.png"
    AUGMENTATION_PLOT = OUTPUT_DIR / "augmentation_samples.png"
    REPORT_PATH = OUTPUT_DIR / "classification_report.txt"

    CLASS_NAMES = ["No Mask", "With Mask"]
    RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------
def generate_face_features(num_samples, label, rng):
    """Generate synthetic image data simulating face images.

    Creates 96x96x3 images with patterns that differ between
    masked and unmasked faces:
    - Skin-tone base with face-like oval structure
    - Mask samples have a distinct lower-face region pattern
    - Added texture, noise, and variation for realism

    Args:
        num_samples: Number of images to generate.
        label: 0 for no mask, 1 for with mask.
        rng: NumPy RandomState for reproducibility.

    Returns:
        Array of shape (num_samples, 96, 96, 3) with values in [0, 1].
    """
    images = np.zeros((num_samples, Config.IMG_SIZE, Config.IMG_SIZE, Config.IMG_CHANNELS),
                      dtype=np.float32)

    for i in range(num_samples):
        img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, Config.IMG_CHANNELS), dtype=np.float32)

        # Background with slight variation
        bg_color = rng.uniform(0.1, 0.3, size=3)
        img[:, :] = bg_color

        # Face-like oval
        center_x, center_y = 48 + rng.randint(-5, 5), 45 + rng.randint(-5, 5)
        face_rx, face_ry = 28 + rng.randint(-3, 3), 35 + rng.randint(-3, 3)

        # Skin tone variation
        skin_base = rng.uniform(0.55, 0.85)
        skin_color = np.array([skin_base, skin_base * 0.8, skin_base * 0.65])

        yy, xx = np.mgrid[:Config.IMG_SIZE, :Config.IMG_SIZE]
        face_mask = ((xx - center_x) / face_rx) ** 2 + ((yy - center_y) / face_ry) ** 2 <= 1

        for c in range(3):
            img[:, :, c] = np.where(face_mask, skin_color[c], img[:, :, c])

        # Eyes region (darker spots in upper face)
        for eye_x_offset in [-12, 12]:
            eye_cx = center_x + eye_x_offset + rng.randint(-2, 2)
            eye_cy = center_y - 10 + rng.randint(-2, 2)
            eye_mask = ((xx - eye_cx) / 5) ** 2 + ((yy - eye_cy) / 3) ** 2 <= 1
            for c in range(3):
                img[:, :, c] = np.where(eye_mask, rng.uniform(0.1, 0.3), img[:, :, c])

        if label == 1:
            # MASK: distinct pattern in lower face
            mask_top = center_y + rng.randint(-2, 5)
            mask_bottom = center_y + face_ry - rng.randint(0, 5)
            mask_color = rng.choice([
                np.array([0.3, 0.5, 0.8]),   # Blue surgical
                np.array([0.9, 0.9, 0.9]),   # White
                np.array([0.2, 0.2, 0.2]),   # Black
                np.array([0.4, 0.7, 0.4]),   # Green
            ])
            mask_region = (
                (yy >= mask_top) & (yy <= mask_bottom) &
                (((xx - center_x) / (face_rx + 2)) ** 2 +
                 ((yy - center_y) / (face_ry + 2)) ** 2 <= 1)
            )
            for c in range(3):
                img[:, :, c] = np.where(mask_region, mask_color[c], img[:, :, c])

            # Mask edges / ear loops (subtle lines)
            for side in [-1, 1]:
                loop_x = center_x + side * (face_rx + 3)
                loop_mask = (np.abs(xx - loop_x) <= 1) & (yy >= mask_top - 5) & (yy <= mask_top + 10)
                for c in range(3):
                    img[:, :, c] = np.where(loop_mask, mask_color[c] * 0.7, img[:, :, c])
        else:
            # NO MASK: mouth and nose features visible
            # Nose
            nose_mask = ((xx - center_x) / 4) ** 2 + ((yy - (center_y + 5)) / 8) ** 2 <= 1
            for c in range(3):
                img[:, :, c] = np.where(nose_mask, skin_color[c] * 0.9, img[:, :, c])

            # Mouth
            mouth_cy = center_y + 18 + rng.randint(-2, 2)
            mouth_mask = (
                ((xx - center_x) / 10) ** 2 + ((yy - mouth_cy) / 3) ** 2 <= 1
            )
            mouth_color = np.array([0.7, 0.35, 0.35])
            for c in range(3):
                img[:, :, c] = np.where(mouth_mask, mouth_color[c], img[:, :, c])

        # Add noise for texture
        noise = rng.randn(Config.IMG_SIZE, Config.IMG_SIZE, Config.IMG_CHANNELS) * 0.03
        img = np.clip(img + noise, 0, 1)

        # Random brightness variation
        brightness = rng.uniform(0.8, 1.2)
        img = np.clip(img * brightness, 0, 1)

        images[i] = img

    return images


def generate_synthetic_dataset():
    """Generate the complete synthetic face mask dataset.

    Returns:
        Tuple of (images, labels) where images are (N, 96, 96, 3)
        and labels are binary (0 = no mask, 1 = with mask).
    """
    print("=" * 60)
    print("Generating Synthetic Face Mask Dataset")
    print("=" * 60)

    rng = np.random.RandomState(Config.RANDOM_SEED)
    half = Config.NUM_SAMPLES // 2

    no_mask_images = generate_face_features(half, label=0, rng=rng)
    mask_images = generate_face_features(half, label=1, rng=rng)

    images = np.concatenate([no_mask_images, mask_images], axis=0)
    labels = np.concatenate([np.zeros(half), np.ones(half)]).astype(np.int32)

    # Shuffle
    idx = rng.permutation(len(labels))
    images = images[idx]
    labels = labels[idx]

    print(f"  Total samples  : {len(labels):,}")
    print(f"  Image shape    : {images.shape[1:]}")
    print(f"  No Mask        : {(labels == 0).sum():,}")
    print(f"  With Mask      : {(labels == 1).sum():,}")
    print()

    return images, labels


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------
def create_augmentation_generator():
    """Create ImageDataGenerator with comprehensive augmentation.

    Returns:
        Configured ImageDataGenerator instance.
    """
    return ImageDataGenerator(
        rotation_range=Config.ROTATION_RANGE,
        width_shift_range=Config.WIDTH_SHIFT,
        height_shift_range=Config.HEIGHT_SHIFT,
        horizontal_flip=Config.HORIZONTAL_FLIP,
        zoom_range=Config.ZOOM_RANGE,
        brightness_range=Config.BRIGHTNESS_RANGE,
        fill_mode="nearest",
    )


def visualize_augmentation(images, labels):
    """Visualize original and augmented image samples.

    Saves a grid showing original images and their augmented versions.
    """
    datagen = create_augmentation_generator()
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    for col in range(6):
        idx = col
        original = images[idx]
        label_name = Config.CLASS_NAMES[labels[idx]]

        # Original
        axes[0, col].imshow(original)
        axes[0, col].set_title(f"Original\n({label_name})", fontsize=10)
        axes[0, col].axis("off")

        # Augmented
        aug_img = datagen.random_transform(original)
        axes[1, col].imshow(np.clip(aug_img, 0, 1))
        axes[1, col].set_title("Augmented", fontsize=10)
        axes[1, col].axis("off")

    plt.suptitle("Data Augmentation Examples", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(Config.AUGMENTATION_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Augmentation samples saved to: {Config.AUGMENTATION_PLOT}")


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
def build_transfer_learning_model():
    """Build a MobileNetV2-based transfer learning model.

    Phase 1: Freeze MobileNetV2 backbone, train only the custom head.
    Phase 2: Unfreeze top layers of backbone for fine-tuning.

    Architecture:
        MobileNetV2 (frozen backbone, ImageNet weights)
        -> GlobalAveragePooling2D
        -> Dense(256, relu) -> BatchNorm -> Dropout(0.5)
        -> Dense(128, relu) -> BatchNorm -> Dropout(0.3)
        -> Dense(1, sigmoid)

    Returns:
        Compiled Keras model with frozen backbone.
    """
    # Load MobileNetV2 backbone
    base_model = keras.applications.MobileNetV2(
        input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, Config.IMG_CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze backbone

    # Custom classification head
    inputs = layers.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, Config.IMG_CHANNELS))

    # Preprocess for MobileNetV2 (expects [-1, 1] range)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="head_dense1")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(0.5, name="head_drop1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense2")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(0.3, name="head_drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs, outputs, name="MaskDetector_MobileNetV2")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=Config.INITIAL_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def unfreeze_top_layers(model, base_model, num_layers_to_unfreeze=30):
    """Unfreeze the top layers of the backbone for fine-tuning.

    Args:
        model: The full model.
        base_model: The MobileNetV2 backbone.
        num_layers_to_unfreeze: Number of top layers to unfreeze.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=Config.FINE_TUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"  Fine-tuning: unfroze top {num_layers_to_unfreeze} backbone layers")
    print(f"  Trainable layers: {trainable_count}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def get_callbacks(phase="phase1"):
    """Create training callbacks.

    Args:
        phase: "phase1" (head training) or "phase2" (fine-tuning).

    Returns:
        List of Keras callbacks.
    """
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patience_es = 7 if phase == "phase1" else 5
    patience_lr = 3

    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(Config.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_model(model, base_model, x_train, y_train, x_val, y_val):
    """Train the model in two phases: head training, then fine-tuning.

    Args:
        model: Compiled Keras model.
        base_model: MobileNetV2 backbone reference.
        x_train: Training images.
        y_train: Training labels.
        x_val: Validation images.
        y_val: Validation labels.

    Returns:
        Combined history from both training phases.
    """
    datagen = create_augmentation_generator()
    datagen.fit(x_train)
    steps_per_epoch = len(x_train) // Config.BATCH_SIZE

    # --- Phase 1: Train classification head ---
    print("=" * 60)
    print("Phase 1: Training Classification Head (backbone frozen)")
    print("=" * 60)

    history1 = model.fit(
        datagen.flow(x_train, y_train, batch_size=Config.BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        epochs=Config.EPOCHS_PHASE1,
        validation_data=(x_val, y_val),
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )

    # --- Phase 2: Fine-tune top backbone layers ---
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning Top Backbone Layers")
    print("=" * 60)

    unfreeze_top_layers(model, base_model, num_layers_to_unfreeze=30)

    history2 = model.fit(
        datagen.flow(x_train, y_train, batch_size=Config.BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        epochs=Config.EPOCHS_PHASE2,
        validation_data=(x_val, y_val),
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )

    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]

    print(f"\n  Training complete. Model saved to: {Config.MODEL_PATH}\n")
    return combined_history


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_training_history(history):
    """Plot training/validation loss and accuracy across both phases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss"]) + 1)
    phase1_end = len(history["loss"]) - Config.EPOCHS_PHASE2

    # Loss
    axes[0].plot(epochs, history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    if phase1_end > 0:
        axes[0].axvline(x=phase1_end, color="red", linestyle="--", alpha=0.7,
                        label="Fine-tuning starts")
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy", linewidth=2)
    if phase1_end > 0:
        axes[1].axvline(x=phase1_end, color="red", linestyle="--", alpha=0.7,
                        label="Fine-tuning starts")
    axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.HISTORY_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training history saved to: {Config.HISTORY_PLOT}")


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=Config.CLASS_NAMES,
        yticklabels=Config.CLASS_NAMES,
        ax=ax, annot_kws={"size": 16},
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(Config.CONFUSION_MATRIX_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to: {Config.CONFUSION_MATRIX_PLOT}")


def plot_roc_and_pr_curves(y_true, y_pred_probs):
    """Plot ROC and Precision-Recall curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_title("ROC Curve", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, linewidth=2, label=f"PR (AUC = {pr_auc:.4f})")
    axes[1].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.ROC_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC & PR curves saved to: {Config.ROC_PLOT}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test):
    """Evaluate the model on the test set.

    Args:
        model: Trained Keras model.
        x_test: Test images.
        y_test: Binary test labels.

    Returns:
        Tuple of (y_true, y_pred, y_pred_probs).
    """
    print("=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print()

    y_pred_probs = model.predict(x_test, verbose=0).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    report = classification_report(
        y_test, y_pred,
        target_names=Config.CLASS_NAMES,
        digits=4,
    )
    print("  Classification Report:")
    print("-" * 60)
    print(report)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(Config.REPORT_PATH, "w") as f:
        f.write("Face Mask Detection - Transfer Learning Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Accuracy : {test_acc:.4f}\n\n")
        f.write(report)
    print(f"  Report saved to: {Config.REPORT_PATH}\n")

    return y_test, y_pred, y_pred_probs


def display_sample_predictions(model, x_test, y_test, num_samples=8):
    """Display sample predictions with confidence scores."""
    print("=" * 60)
    print(f"Sample Predictions ({num_samples} random test images)")
    print("=" * 60)

    rng = np.random.RandomState(Config.RANDOM_SEED)
    indices = rng.choice(len(x_test), num_samples, replace=False)
    predictions = model.predict(x_test[indices], verbose=0).flatten()

    for i, idx in enumerate(indices):
        true_label = Config.CLASS_NAMES[y_test[idx]]
        prob = predictions[i]
        pred_label = Config.CLASS_NAMES[int(prob >= 0.5)]
        confidence = prob if prob >= 0.5 else 1 - prob
        status = "CORRECT" if true_label == pred_label else "WRONG"

        print(f"  [{i+1:>2}] True: {true_label:<10} | Predicted: {pred_label:<10} "
              f"| Confidence: {confidence:.2%} | {status}")
    print()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    """Execute the full face mask detection pipeline."""
    print("\n" + "=" * 60)
    print("  Face Mask Detection with Transfer Learning")
    print("=" * 60 + "\n")

    tf.random.set_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic data
    images, labels = generate_synthetic_dataset()

    # 2. Visualize augmentation
    visualize_augmentation(images, labels)

    # 3. Split data
    x_train, x_temp, y_train, y_temp = train_test_split(
        images, labels,
        test_size=Config.VALIDATION_SPLIT + Config.TEST_SPLIT,
        random_state=Config.RANDOM_SEED,
        stratify=labels,
    )
    test_ratio = Config.TEST_SPLIT / (Config.VALIDATION_SPLIT + Config.TEST_SPLIT)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp,
        test_size=test_ratio,
        random_state=Config.RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"  Train : {len(y_train):,}  |  Val : {len(y_val):,}  |  Test : {len(y_test):,}")
    print()

    # 4. Build model
    model, base_model = build_transfer_learning_model()
    print("=" * 60)
    print("Model Architecture")
    print("=" * 60)
    model.summary()
    print(f"\n  Total parameters: {model.count_params():,}\n")

    # 5. Train (two-phase: head -> fine-tune)
    history = train_model(model, base_model, x_train, y_train, x_val, y_val)

    # 6. Plot training history
    plot_training_history(history)

    # 7. Evaluate
    y_true, y_pred, y_pred_probs = evaluate_model(model, x_test, y_test)

    # 8. Confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # 9. ROC and PR curves
    plot_roc_and_pr_curves(y_true, y_pred_probs)

    # 10. Sample predictions
    display_sample_predictions(model, x_test, y_test)

    # 11. Save final model
    model.save(Config.MODEL_PATH)
    print(f"  Final model saved to: {Config.MODEL_PATH}")

    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
