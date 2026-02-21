"""
Music Genre Classification with Deep Neural Networks
=====================================================

A production-quality pipeline for classifying music tracks into 10 genres
using synthesized audio feature data (MFCCs, spectral features, chroma, tempo).

Architecture:
    - CNN branch for MFCC spectral features
    - DNN branch for hand-crafted audio features
    - Multi-input fusion with concatenation
    - Comprehensive evaluation with confusion matrix and feature importance

Genres:
    Rock, Pop, Jazz, Classical, Hip-Hop, Country, Metal, Reggae, Blues, Electronic

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    """Central configuration for the genre classification pipeline."""

    # Data
    NUM_SAMPLES = 10000
    NUM_GENRES = 10
    GENRE_NAMES = [
        "Blues", "Classical", "Country", "Electronic", "Hip-Hop",
        "Jazz", "Metal", "Pop", "Reggae", "Rock",
    ]

    # Audio feature dimensions
    NUM_MFCC_COEFFICIENTS = 13
    MFCC_TIME_STEPS = 130  # Simulated time frames
    NUM_SCALAR_FEATURES = 26  # Hand-crafted features

    # Training
    BATCH_SIZE = 64
    EPOCHS = 60
    INITIAL_LR = 1e-3
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # Paths
    OUTPUT_DIR = Path("outputs")
    MODEL_PATH = OUTPUT_DIR / "genre_classifier_model.keras"
    HISTORY_PLOT = OUTPUT_DIR / "training_history.png"
    CONFUSION_MATRIX_PLOT = OUTPUT_DIR / "confusion_matrix.png"
    FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "feature_importance.png"
    REPORT_PATH = OUTPUT_DIR / "classification_report.txt"

    RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------
def generate_genre_profiles():
    """Define characteristic audio feature profiles for each genre.

    Each genre has a distinct statistical fingerprint in terms of:
    - MFCC patterns (timbre/spectral shape)
    - Spectral centroid, bandwidth, rolloff (brightness)
    - Chroma features (harmonic content)
    - Tempo, zero crossing rate, RMS energy

    Returns:
        Dictionary mapping genre index to feature profile dict.
    """
    profiles = {
        0: {  # Blues
            "tempo_mean": 95, "tempo_std": 15,
            "spectral_centroid": 2200, "spectral_bw": 1800,
            "zcr": 0.06, "rms": 0.15, "rolloff": 3800,
            "mfcc_base": np.array([1, -5, 3, 2, -1, 0.5, -0.5, 1, -1, 0.5, 0, -0.5, 0.3]),
            "chroma_emphasis": [0, 4, 7],  # Major/minor blues
        },
        1: {  # Classical
            "tempo_mean": 110, "tempo_std": 35,
            "spectral_centroid": 1800, "spectral_bw": 2200,
            "zcr": 0.04, "rms": 0.10, "rolloff": 3200,
            "mfcc_base": np.array([2, -3, 5, -2, 3, -1, 2, -1.5, 1, -0.5, 1, -0.3, 0.5]),
            "chroma_emphasis": [0, 4, 7],
        },
        2: {  # Country
            "tempo_mean": 120, "tempo_std": 15,
            "spectral_centroid": 2600, "spectral_bw": 1600,
            "zcr": 0.07, "rms": 0.18, "rolloff": 4200,
            "mfcc_base": np.array([0.5, -4, 2.5, 1.5, -0.5, 1, 0, 0.5, -0.5, 0, 0.3, -0.2, 0.1]),
            "chroma_emphasis": [0, 5, 7],
        },
        3: {  # Electronic
            "tempo_mean": 128, "tempo_std": 10,
            "spectral_centroid": 3500, "spectral_bw": 2500,
            "zcr": 0.09, "rms": 0.25, "rolloff": 5500,
            "mfcc_base": np.array([-1, -7, 1, -1, 2, -2, 1, -1, 0.5, -1, 0.5, -0.5, 0.2]),
            "chroma_emphasis": [0, 3, 7],
        },
        4: {  # Hip-Hop
            "tempo_mean": 90, "tempo_std": 12,
            "spectral_centroid": 2000, "spectral_bw": 2000,
            "zcr": 0.08, "rms": 0.22, "rolloff": 3500,
            "mfcc_base": np.array([-2, -8, 0, -2, 1, -1, 0, -0.5, 0.5, -0.5, 0, -0.3, 0.1]),
            "chroma_emphasis": [0, 3, 5],
        },
        5: {  # Jazz
            "tempo_mean": 135, "tempo_std": 30,
            "spectral_centroid": 2400, "spectral_bw": 2100,
            "zcr": 0.05, "rms": 0.12, "rolloff": 4000,
            "mfcc_base": np.array([1.5, -4, 4, -1, 2, 0, 1.5, -1, 1, -0.3, 0.8, -0.2, 0.4]),
            "chroma_emphasis": [0, 4, 7, 10],  # Extended chords
        },
        6: {  # Metal
            "tempo_mean": 150, "tempo_std": 25,
            "spectral_centroid": 3800, "spectral_bw": 2800,
            "zcr": 0.12, "rms": 0.30, "rolloff": 6000,
            "mfcc_base": np.array([-3, -10, -1, -3, 0, -2, -1, -1.5, 0, -1, -0.5, -0.8, -0.3]),
            "chroma_emphasis": [0, 1, 7],  # Power chords + dissonance
        },
        7: {  # Pop
            "tempo_mean": 118, "tempo_std": 12,
            "spectral_centroid": 2800, "spectral_bw": 1500,
            "zcr": 0.07, "rms": 0.20, "rolloff": 4500,
            "mfcc_base": np.array([0, -5, 2, 0, 1, 0, 0.5, 0, 0.5, 0, 0.2, 0, 0.1]),
            "chroma_emphasis": [0, 4, 7],
        },
        8: {  # Reggae
            "tempo_mean": 80, "tempo_std": 10,
            "spectral_centroid": 2100, "spectral_bw": 1400,
            "zcr": 0.05, "rms": 0.16, "rolloff": 3400,
            "mfcc_base": np.array([0.8, -6, 1, 1, 0, 0.5, -0.3, 0.8, -0.3, 0.3, 0.1, -0.1, 0.2]),
            "chroma_emphasis": [0, 3, 7],
        },
        9: {  # Rock
            "tempo_mean": 130, "tempo_std": 20,
            "spectral_centroid": 3200, "spectral_bw": 2300,
            "zcr": 0.10, "rms": 0.26, "rolloff": 5200,
            "mfcc_base": np.array([-1.5, -8, 0.5, -1.5, 1, -1, 0, -0.8, 0.3, -0.5, -0.2, -0.4, 0]),
            "chroma_emphasis": [0, 5, 7],  # Power chords
        },
    }
    return profiles


def generate_synthetic_data():
    """Generate synthetic audio feature data for all genres.

    For each sample we produce:
    1. MFCC matrix (13 coefficients x 130 time frames) -- CNN input
    2. Scalar feature vector (26 features) -- DNN input

    Scalar features:
        - Tempo (1)
        - Spectral: centroid, bandwidth, rolloff, contrast, flatness (5)
        - Chroma: 12 pitch classes (12)
        - Zero crossing rate (1)
        - RMS energy (1)
        - MFCC means (13) -> we reduce to a few aggregate stats (6)

    Returns:
        Tuple of (mfcc_data, scalar_data, labels)
        mfcc_data:   shape (N, 130, 13, 1)  -- CNN input
        scalar_data: shape (N, 26)           -- DNN input
        labels:      shape (N,)              -- integer labels
    """
    print("=" * 60)
    print("Generating Synthetic Audio Feature Data")
    print("=" * 60)

    rng = np.random.RandomState(Config.RANDOM_SEED)
    profiles = generate_genre_profiles()

    samples_per_genre = Config.NUM_SAMPLES // Config.NUM_GENRES

    mfcc_list = []
    scalar_list = []
    label_list = []

    for genre_idx, profile in profiles.items():
        for _ in range(samples_per_genre):
            # --- MFCC matrix ---
            base = profile["mfcc_base"]
            # Create time-varying MFCCs by adding temporal structure
            mfcc = np.tile(base, (Config.MFCC_TIME_STEPS, 1))
            # Add temporal variation (slow drift + noise)
            temporal_drift = np.sin(
                np.linspace(0, 2 * np.pi, Config.MFCC_TIME_STEPS)
            ).reshape(-1, 1) * rng.randn(1, Config.NUM_MFCC_COEFFICIENTS) * 0.5
            noise = rng.randn(Config.MFCC_TIME_STEPS, Config.NUM_MFCC_COEFFICIENTS) * 1.5
            mfcc = mfcc + temporal_drift + noise

            # --- Scalar features ---
            tempo = rng.normal(profile["tempo_mean"], profile["tempo_std"])
            spectral_centroid = rng.normal(profile["spectral_centroid"], 200)
            spectral_bw = rng.normal(profile["spectral_bw"], 200)
            spectral_rolloff = rng.normal(profile["rolloff"], 300)
            spectral_contrast = rng.normal(25, 5)
            spectral_flatness = rng.normal(0.1, 0.03)

            # Chroma features (12 pitch classes)
            chroma = rng.uniform(0.2, 0.5, size=12)
            for note in profile["chroma_emphasis"]:
                chroma[note % 12] += rng.uniform(0.2, 0.4)
            chroma = chroma / chroma.sum()  # Normalize

            zcr = rng.normal(profile["zcr"], 0.01)
            rms = rng.normal(profile["rms"], 0.03)

            # Aggregate MFCC stats
            mfcc_mean = mfcc.mean(axis=0)[:3]
            mfcc_std = mfcc.std(axis=0)[:3]

            scalar = np.concatenate([
                [tempo, spectral_centroid, spectral_bw, spectral_rolloff,
                 spectral_contrast, spectral_flatness],
                chroma,
                [zcr, rms],
                mfcc_mean, mfcc_std,
            ])

            mfcc_list.append(mfcc)
            scalar_list.append(scalar)
            label_list.append(genre_idx)

    mfcc_data = np.array(mfcc_list)[..., np.newaxis]  # Add channel dim
    scalar_data = np.array(scalar_list)
    labels = np.array(label_list)

    # Shuffle
    shuffle_idx = rng.permutation(len(labels))
    mfcc_data = mfcc_data[shuffle_idx]
    scalar_data = scalar_data[shuffle_idx]
    labels = labels[shuffle_idx]

    print(f"  Total samples     : {len(labels):,}")
    print(f"  MFCC shape        : {mfcc_data.shape[1:]} (time x coefficients x channels)")
    print(f"  Scalar features   : {scalar_data.shape[1]}")
    print(f"  Classes           : {Config.NUM_GENRES}")
    print(f"  Samples per genre : {samples_per_genre}")
    print()

    return mfcc_data, scalar_data, labels


def augment_audio_features(mfcc_data, scalar_data, labels, augment_factor=2):
    """Augment audio features by adding noise and time-shifting.

    Args:
        mfcc_data: MFCC array, shape (N, T, C, 1).
        scalar_data: Scalar features, shape (N, F).
        labels: Integer labels, shape (N,).
        augment_factor: How many augmented copies to create.

    Returns:
        Augmented (mfcc_data, scalar_data, labels).
    """
    print("  Applying data augmentation...")
    rng = np.random.RandomState(Config.RANDOM_SEED + 1)

    aug_mfcc = [mfcc_data]
    aug_scalar = [scalar_data]
    aug_labels = [labels]

    for _ in range(augment_factor - 1):
        # Noise injection
        noisy_mfcc = mfcc_data + rng.randn(*mfcc_data.shape) * 0.3

        # Time shift (roll along time axis)
        shift = rng.randint(-10, 10, size=len(mfcc_data))
        shifted_mfcc = np.array([
            np.roll(m, s, axis=0) for m, s in zip(noisy_mfcc, shift)
        ])

        # Slight scalar perturbation
        noisy_scalar = scalar_data + rng.randn(*scalar_data.shape) * 0.05 * np.abs(scalar_data)

        aug_mfcc.append(shifted_mfcc)
        aug_scalar.append(noisy_scalar)
        aug_labels.append(labels)

    mfcc_aug = np.concatenate(aug_mfcc, axis=0)
    scalar_aug = np.concatenate(aug_scalar, axis=0)
    labels_aug = np.concatenate(aug_labels, axis=0)

    # Shuffle
    idx = rng.permutation(len(labels_aug))
    print(f"  Augmented dataset : {len(labels_aug):,} samples ({augment_factor}x)\n")
    return mfcc_aug[idx], scalar_aug[idx], labels_aug[idx]


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
def build_genre_model():
    """Build a multi-input CNN+DNN model for genre classification.

    Architecture:
        CNN branch (MFCC input):
            Conv2D(32) -> BN -> MaxPool
            Conv2D(64) -> BN -> MaxPool
            Conv2D(128) -> BN -> GlobalAveragePool -> Dropout

        DNN branch (scalar input):
            Dense(128) -> BN -> Dropout
            Dense(64) -> BN -> Dropout

        Fusion:
            Concatenate -> Dense(128) -> Dropout -> Dense(10, softmax)

    Returns:
        Compiled Keras model.
    """
    # --- CNN branch for MFCC ---
    mfcc_input = layers.Input(
        shape=(Config.MFCC_TIME_STEPS, Config.NUM_MFCC_COEFFICIENTS, 1),
        name="mfcc_input",
    )
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="cnn_conv1")(mfcc_input)
    x = layers.BatchNormalization(name="cnn_bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_pool1")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="cnn_conv2")(x)
    x = layers.BatchNormalization(name="cnn_bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_pool2")(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="cnn_conv3")(x)
    x = layers.BatchNormalization(name="cnn_bn3")(x)
    x = layers.GlobalAveragePooling2D(name="cnn_gap")(x)
    x = layers.Dropout(0.3, name="cnn_dropout")(x)
    cnn_out = x

    # --- DNN branch for scalar features ---
    scalar_input = layers.Input(
        shape=(Config.NUM_SCALAR_FEATURES,),
        name="scalar_input",
    )
    y = layers.Dense(128, activation="relu", name="dnn_dense1")(scalar_input)
    y = layers.BatchNormalization(name="dnn_bn1")(y)
    y = layers.Dropout(0.3, name="dnn_drop1")(y)

    y = layers.Dense(64, activation="relu", name="dnn_dense2")(y)
    y = layers.BatchNormalization(name="dnn_bn2")(y)
    y = layers.Dropout(0.3, name="dnn_drop2")(y)
    dnn_out = y

    # --- Fusion ---
    merged = layers.Concatenate(name="fusion")([cnn_out, dnn_out])
    z = layers.Dense(128, activation="relu", name="fusion_dense1")(merged)
    z = layers.BatchNormalization(name="fusion_bn")(z)
    z = layers.Dropout(0.4, name="fusion_drop")(z)
    output = layers.Dense(Config.NUM_GENRES, activation="softmax", name="output")(z)

    model = models.Model(
        inputs=[mfcc_input, scalar_input],
        outputs=output,
        name="Genre_Classifier",
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=Config.INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, train_data, val_data):
    """Train the genre classifier.

    Args:
        model: Compiled Keras model.
        train_data: Tuple of ([mfcc_train, scalar_train], y_train).
        val_data: Tuple of ([mfcc_val, scalar_val], y_val).

    Returns:
        Keras History object.
    """
    print("=" * 60)
    print("Training Model")
    print("=" * 60)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(Config.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]

    history = model.fit(
        train_data[0], train_data[1],
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=val_data,
        callbacks=cb_list,
        verbose=1,
    )

    print(f"\n  Training complete. Best model saved to: {Config.MODEL_PATH}\n")
    return history


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_training_history(history):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

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
    print(f"  Training history saved to: {Config.HISTORY_PLOT}")


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix heatmap for genre classification."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=Config.GENRE_NAMES, yticklabels=Config.GENRE_NAMES,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted Genre")
    axes[0].set_ylabel("True Genre")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=Config.GENRE_NAMES, yticklabels=Config.GENRE_NAMES,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted Genre")
    axes[1].set_ylabel("True Genre")

    plt.tight_layout()
    plt.savefig(Config.CONFUSION_MATRIX_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to: {Config.CONFUSION_MATRIX_PLOT}")


def plot_feature_importance(model, scalar_data, labels, feature_names):
    """Estimate and plot feature importance via permutation.

    Permutes each scalar feature and measures the drop in accuracy
    to estimate its importance.

    Args:
        model: Trained model.
        scalar_data: Scalar test features, shape (N, F).
        labels: Integer test labels, shape (N,).
        feature_names: List of feature name strings.
    """
    print("\n  Computing feature importance (permutation-based)...")

    # Use a subset for speed
    rng = np.random.RandomState(Config.RANDOM_SEED)
    n_samples = min(1000, len(labels))
    idx = rng.choice(len(labels), n_samples, replace=False)
    scalar_sub = scalar_data[idx]
    labels_sub = labels[idx]

    # Need dummy MFCC input -- generate zeros
    mfcc_dummy = np.zeros(
        (n_samples, Config.MFCC_TIME_STEPS, Config.NUM_MFCC_COEFFICIENTS, 1),
        dtype="float32",
    )

    # We estimate importance of scalar features only
    # Baseline accuracy using only scalar features (with zero MFCC)
    # This isolates scalar branch importance
    baseline_preds = model.predict([mfcc_dummy, scalar_sub], verbose=0)
    baseline_acc = np.mean(np.argmax(baseline_preds, axis=1) == labels_sub)

    importance = []
    for feat_idx in range(scalar_data.shape[1]):
        shuffled = scalar_sub.copy()
        shuffled[:, feat_idx] = rng.permutation(shuffled[:, feat_idx])
        perm_preds = model.predict([mfcc_dummy, shuffled], verbose=0)
        perm_acc = np.mean(np.argmax(perm_preds, axis=1) == labels_sub)
        importance.append(baseline_acc - perm_acc)

    importance = np.array(importance)

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    top_n = min(15, len(feature_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    y_pos = np.arange(top_n)

    ax.barh(
        y_pos,
        importance[sorted_idx[:top_n]][::-1],
        color=colors,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]][::-1])
    ax.set_xlabel("Accuracy Drop (Permutation Importance)")
    ax.set_title("Feature Importance (Top Scalar Features)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(Config.FEATURE_IMPORTANCE_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot saved to: {Config.FEATURE_IMPORTANCE_PLOT}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, test_data, y_test):
    """Evaluate the model on the test set.

    Args:
        model: Trained Keras model.
        test_data: List of [mfcc_test, scalar_test].
        y_test: Integer test labels.

    Returns:
        Tuple of (y_true, y_pred).
    """
    print("=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(test_data, y_test, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print()

    y_pred_probs = model.predict(test_data, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_test, y_pred,
        target_names=Config.GENRE_NAMES,
        digits=4,
    )
    print("  Classification Report:")
    print("-" * 60)
    print(report)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(Config.REPORT_PATH, "w") as f:
        f.write("Music Genre Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Accuracy : {test_acc:.4f}\n\n")
        f.write(report)
    print(f"  Report saved to: {Config.REPORT_PATH}\n")

    return y_test, y_pred


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    """Execute the full genre classification pipeline."""
    print("\n" + "=" * 60)
    print("  Music Genre Classification with CNN + DNN")
    print("=" * 60 + "\n")

    tf.random.set_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic data
    mfcc_data, scalar_data, labels = generate_synthetic_data()

    # 2. Augment
    mfcc_data, scalar_data, labels = augment_audio_features(
        mfcc_data, scalar_data, labels, augment_factor=2,
    )

    # 3. Scale scalar features
    scaler = StandardScaler()
    scalar_data = scaler.fit_transform(scalar_data)

    # 4. Split data
    (mfcc_train, mfcc_temp, scalar_train, scalar_temp,
     y_train, y_temp) = train_test_split(
        mfcc_data, scalar_data, labels,
        test_size=Config.VALIDATION_SPLIT + Config.TEST_SPLIT,
        random_state=Config.RANDOM_SEED, stratify=labels,
    )
    test_ratio = Config.TEST_SPLIT / (Config.VALIDATION_SPLIT + Config.TEST_SPLIT)
    (mfcc_val, mfcc_test, scalar_val, scalar_test,
     y_val, y_test) = train_test_split(
        mfcc_temp, scalar_temp, y_temp,
        test_size=test_ratio,
        random_state=Config.RANDOM_SEED, stratify=y_temp,
    )

    print(f"  Train : {len(y_train):,}  |  Val : {len(y_val):,}  |  Test : {len(y_test):,}")
    print()

    # 5. Build model
    model = build_genre_model()
    print("=" * 60)
    print("Model Architecture")
    print("=" * 60)
    model.summary()
    print(f"\n  Total parameters: {model.count_params():,}\n")

    # 6. Train
    history = train_model(
        model,
        train_data=([mfcc_train, scalar_train], y_train),
        val_data=([mfcc_val, scalar_val], y_val),
    )

    # 7. Visualize training
    plot_training_history(history)

    # 8. Evaluate
    y_true, y_pred = evaluate_model(model, [mfcc_test, scalar_test], y_test)

    # 9. Confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # 10. Feature importance
    feature_names = (
        ["Tempo", "Spectral Centroid", "Spectral BW", "Spectral Rolloff",
         "Spectral Contrast", "Spectral Flatness"]
        + [f"Chroma_{i}" for i in range(12)]
        + ["ZCR", "RMS"]
        + [f"MFCC_mean_{i}" for i in range(3)]
        + [f"MFCC_std_{i}" for i in range(3)]
    )
    plot_feature_importance(model, scalar_test, y_test, feature_names)

    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
