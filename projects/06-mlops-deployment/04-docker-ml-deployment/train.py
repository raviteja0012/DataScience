"""
Model Training Script for Containerised Deployment
====================================================
Trains a scikit-learn classifier, evaluates it, and persists the model
artefact along with metadata to disk so that the FastAPI serving layer
can load it at startup.

Usage:
    python train.py                           # defaults (random_forest, iris)
    python train.py --algorithm gradient_boosting
    python train.py --output /app/models
"""

import os
import json
import pickle
import hashlib
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}

ALGORITHMS = {
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=100, random_state=seed
    ),
    "gradient_boosting": lambda seed: GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=seed
    ),
    "logistic_regression": lambda seed: LogisticRegression(
        max_iter=1000, random_state=seed
    ),
}


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_and_save(
    algorithm: str = "random_forest",
    dataset: str = "iris",
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = None,
):
    """Full training routine: load -> split -> train -> evaluate -> save."""

    output_dir = Path(output_dir or Path(__file__).resolve().parent / "models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    loader = DATASETS[dataset]
    raw = loader()
    feature_names = [f"feature_{i}" for i in range(raw.data.shape[1])]
    target_names = list(raw.target_names) if hasattr(raw, "target_names") else None

    X = pd.DataFrame(raw.data, columns=feature_names)
    y = pd.Series(raw.target, name="target")

    print(f"Dataset:    {dataset} ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f"Algorithm:  {algorithm}")
    print(f"Seed:       {random_state}")

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # ---- Build pipeline ----
    estimator = ALGORITHMS[algorithm](random_state)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])

    # ---- Train ----
    pipeline.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = pipeline.predict(X_test)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
    }

    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        output_dict=True,
    )

    print(f"\nEvaluation Results:")
    print(f"  Accuracy:   {metrics['accuracy']}")
    print(f"  Precision:  {metrics['precision_macro']}")
    print(f"  Recall:     {metrics['recall_macro']}")
    print(f"  F1 Macro:   {metrics['f1_macro']}")
    print(f"  CV Mean:    {metrics['cv_accuracy_mean']} +/- {metrics['cv_accuracy_std']}")

    # ---- Save model ----
    model_path = output_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    # ---- Save metadata ----
    metadata = {
        "algorithm": algorithm,
        "dataset": dataset,
        "random_state": random_state,
        "feature_names": feature_names,
        "target_names": target_names,
        "n_classes": len(np.unique(y)),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": metrics,
        "classification_report": report,
        "model_hash": model_hash,
        "trained_at": datetime.datetime.utcnow().isoformat(),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nModel saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Model hash: {model_hash}")

    return pipeline, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train and persist an ML model")
    parser.add_argument(
        "--algorithm",
        default="random_forest",
        choices=list(ALGORITHMS.keys()),
    )
    parser.add_argument(
        "--dataset",
        default="iris",
        choices=list(DATASETS.keys()),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output directory for model artefacts")
    args = parser.parse_args()

    train_and_save(
        algorithm=args.algorithm,
        dataset=args.dataset,
        test_size=args.test_size,
        random_state=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
