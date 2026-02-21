"""
Sentiment Analysis with Bi-LSTM and Attention Mechanism
=======================================================

A production-quality NLP pipeline for binary sentiment classification
on the IMDB movie review dataset.

Architecture:
    - Embedding layer with learned word vectors
    - Bidirectional LSTM layers for sequential feature extraction
    - Custom attention mechanism to focus on important tokens
    - Dense classification head with dropout

Dataset:
    IMDB Reviews - 50,000 movie reviews (25k train, 25k test)

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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    """Central configuration for the sentiment analysis pipeline."""

    # Data
    VOCAB_SIZE = 20000
    MAX_SEQUENCE_LENGTH = 300
    EMBEDDING_DIM = 128

    # Model
    LSTM_UNITS = 64
    DENSE_UNITS = 64
    DROPOUT_RATE = 0.3

    # Training
    BATCH_SIZE = 64
    EPOCHS = 20
    INITIAL_LR = 1e-3

    # Paths
    OUTPUT_DIR = Path("outputs")
    MODEL_PATH = OUTPUT_DIR / "sentiment_lstm_model.keras"
    HISTORY_PLOT = OUTPUT_DIR / "training_history.png"
    CONFUSION_MATRIX_PLOT = OUTPUT_DIR / "confusion_matrix.png"
    ROC_PLOT = OUTPUT_DIR / "roc_curve.png"
    REPORT_PATH = OUTPUT_DIR / "classification_report.txt"

    CLASS_NAMES = ["Negative", "Positive"]


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------
def load_and_preprocess_data():
    """Load the IMDB dataset and apply tokenization and padding.

    Uses keras.datasets.imdb which provides pre-indexed integer sequences.
    Sequences are padded/truncated to Config.MAX_SEQUENCE_LENGTH.

    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test, word_index)
    """
    print("=" * 60)
    print("Loading IMDB Dataset")
    print("=" * 60)

    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.imdb.load_data(
        num_words=Config.VOCAB_SIZE,
    )

    # Pad sequences to uniform length
    x_train_full = pad_sequences(
        x_train_full,
        maxlen=Config.MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )
    x_test = pad_sequences(
        x_test,
        maxlen=Config.MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    # Split training set into train + validation
    val_size = 5000
    x_val = x_train_full[:val_size]
    y_val = y_train_full[:val_size]
    x_train = x_train_full[val_size:]
    y_train = y_train_full[val_size:]

    # Get word index for decoding / encoding custom text
    word_index = keras.datasets.imdb.get_word_index()

    print(f"  Vocabulary size    : {Config.VOCAB_SIZE:,}")
    print(f"  Max sequence length: {Config.MAX_SEQUENCE_LENGTH}")
    print(f"  Training samples   : {len(x_train):,}")
    print(f"  Validation samples : {len(x_val):,}")
    print(f"  Test samples       : {len(x_test):,}")
    print(f"  Positive ratio     : {y_train.mean():.2%} (train)")
    print()

    return x_train, y_train, x_val, y_val, x_test, y_test, word_index


# ---------------------------------------------------------------------------
# Attention Layer
# ---------------------------------------------------------------------------
class AttentionLayer(layers.Layer):
    """Bahdanau-style additive attention for sequence classification.

    Computes a weighted sum of LSTM hidden states, where weights are
    learned attention scores indicating token importance.

    Attributes:
        W: Dense projection to attention hidden dimension.
        V: Dense projection to scalar attention score.
    """

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = layers.Dense(self.units, activation="tanh", name="attn_W")
        self.V = layers.Dense(1, name="attn_V")
        super().build(input_shape)

    def call(self, inputs):
        """Apply attention mechanism.

        Args:
            inputs: LSTM output tensor, shape (batch, timesteps, features).

        Returns:
            Context vector, shape (batch, features).
        """
        # Score each timestep
        score = self.V(self.W(inputs))               # (batch, timesteps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, timesteps, 1)

        # Weighted sum
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
def build_bilstm_attention_model():
    """Build a Bi-LSTM model with attention for sentiment classification.

    Architecture:
        Embedding(20000, 128)
        -> SpatialDropout1D(0.2)
        -> Bidirectional(LSTM(64, return_sequences=True))
        -> Bidirectional(LSTM(32, return_sequences=True))
        -> AttentionLayer(64)
        -> Dense(64, relu) -> Dropout(0.3)
        -> Dense(1, sigmoid)

    Returns:
        Compiled Keras model.
    """
    inputs = layers.Input(shape=(Config.MAX_SEQUENCE_LENGTH,), name="input_tokens")

    # Embedding
    x = layers.Embedding(
        input_dim=Config.VOCAB_SIZE,
        output_dim=Config.EMBEDDING_DIM,
        input_length=Config.MAX_SEQUENCE_LENGTH,
        name="embedding",
    )(inputs)
    x = layers.SpatialDropout1D(0.2, name="spatial_dropout")(x)

    # Bi-LSTM layers (return_sequences=True for attention)
    x = layers.Bidirectional(
        layers.LSTM(Config.LSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_1",
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(Config.LSTM_UNITS // 2, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_2",
    )(x)

    # Attention
    x = AttentionLayer(units=Config.LSTM_UNITS, name="attention")(x)

    # Classification head
    x = layers.Dense(Config.DENSE_UNITS, activation="relu", name="dense1")(x)
    x = layers.Dropout(Config.DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Attention")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=Config.INITIAL_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def print_model_summary(model):
    """Print formatted model architecture summary."""
    print("=" * 60)
    print("Model Architecture")
    print("=" * 60)
    model.summary()
    print(f"\n  Total parameters: {model.count_params():,}\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def get_callbacks():
    """Create training callbacks.

    Returns:
        List containing EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.
    """
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(Config.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_model(model, x_train, y_train, x_val, y_val):
    """Train the Bi-LSTM attention model.

    Args:
        model: Compiled Keras model.
        x_train: Padded training sequences.
        y_train: Binary training labels.
        x_val: Padded validation sequences.
        y_val: Binary validation labels.

    Returns:
        Keras History object.
    """
    print("=" * 60)
    print("Training Model")
    print("=" * 60)

    history = model.fit(
        x_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=get_callbacks(),
        verbose=1,
    )

    print(f"\n  Training complete. Best model saved to: {Config.MODEL_PATH}\n")
    return history


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_training_history(history):
    """Plot training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
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
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=Config.CLASS_NAMES,
        yticklabels=Config.CLASS_NAMES,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(Config.CONFUSION_MATRIX_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to: {Config.CONFUSION_MATRIX_PLOT}")


def plot_roc_curve(y_true, y_pred_probs):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.ROC_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved to: {Config.ROC_PLOT}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test):
    """Evaluate the model on the test set with detailed metrics.

    Args:
        model: Trained Keras model.
        x_test: Padded test sequences.
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

    # Save report
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(Config.REPORT_PATH, "w") as f:
        f.write("IMDB Sentiment Analysis - Bi-LSTM with Attention\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Accuracy : {test_acc:.4f}\n\n")
        f.write(report)
    print(f"  Report saved to: {Config.REPORT_PATH}")
    print()

    return y_test, y_pred, y_pred_probs


# ---------------------------------------------------------------------------
# Custom Text Prediction
# ---------------------------------------------------------------------------
def predict_custom_text(model, word_index, texts):
    """Predict sentiment for custom user-provided text.

    Args:
        model: Trained Keras model.
        word_index: Dictionary mapping words to integer indices.
        texts: List of raw text strings to classify.
    """
    print("=" * 60)
    print("Custom Text Predictions")
    print("=" * 60)

    # Build reverse word index for encoding
    # IMDB word_index is offset by 3 (0=padding, 1=start, 2=unknown)
    index_offset = 3

    for i, text in enumerate(texts):
        # Tokenize: lowercase, split, convert to indices
        words = text.lower().split()
        sequence = [
            word_index.get(word, 2) + index_offset
            for word in words
        ]
        # Cap indices at vocab size
        sequence = [idx if idx < Config.VOCAB_SIZE else 2 for idx in sequence]

        # Pad
        padded = pad_sequences(
            [sequence],
            maxlen=Config.MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post",
        )

        prob = model.predict(padded, verbose=0)[0][0]
        sentiment = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
        confidence = prob if prob >= 0.5 else 1 - prob

        print(f"\n  [{i+1}] \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        print(f"      Sentiment  : {sentiment}")
        print(f"      Confidence : {confidence:.2%}")
        print(f"      Raw score  : {prob:.4f}")

    print()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    """Execute the full sentiment analysis pipeline."""
    print("\n" + "=" * 60)
    print("  Sentiment Analysis with Bi-LSTM + Attention")
    print("=" * 60 + "\n")

    # Reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and preprocess data
    x_train, y_train, x_val, y_val, x_test, y_test, word_index = (
        load_and_preprocess_data()
    )

    # 2. Build model
    model = build_bilstm_attention_model()
    print_model_summary(model)

    # 3. Train
    history = train_model(model, x_train, y_train, x_val, y_val)

    # 4. Visualize training
    plot_training_history(history)

    # 5. Evaluate
    y_true, y_pred, y_pred_probs = evaluate_model(model, x_test, y_test)

    # 6. Confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # 7. ROC curve
    plot_roc_curve(y_true, y_pred_probs)

    # 8. Custom predictions
    sample_reviews = [
        "This movie was absolutely fantastic! Great acting, brilliant storyline, and amazing cinematography.",
        "Terrible film. Waste of time and money. The plot made no sense and the acting was awful.",
        "An okay movie, not the best but certainly not the worst. Some good moments mixed with boring scenes.",
        "A masterpiece of modern cinema. Every frame is beautiful and the story keeps you on the edge of your seat.",
        "I fell asleep halfway through. Boring dialogue, predictable plot, and forgettable characters.",
    ]
    predict_custom_text(model, word_index, sample_reviews)

    # 9. Save final model
    model.save(Config.MODEL_PATH)
    print(f"  Final model saved to: {Config.MODEL_PATH}")

    print("\n" + "=" * 60)
    print("  Pipeline Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
