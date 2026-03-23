"""
Flask REST API for ML Model Serving
====================================
Production-grade model serving API with:
- Single and batch prediction endpoints
- Input validation and error handling
- Request/response logging with structured output
- Model versioning support
- Health checks and model introspection

Usage:
    python app.py                  # Start the server
    python app.py --train          # Retrain the model then start
    python app.py --port 8080      # Custom port
"""

import os
import json
import time
import uuid
import pickle
import hashlib
import logging
import argparse
import datetime
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, g
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "api.log"),
    ],
)
logger = logging.getLogger("model-api")

# ---------------------------------------------------------------------------
# Model registry – keeps track of all trained model versions
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Simple file-based model registry with versioning."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.registry_path = model_dir / "registry.json"
        self._load_registry()

    def _load_registry(self):
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": [], "active_version": None}

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register(self, model, metadata: dict) -> str:
        """Save a model and register it with metadata."""
        version = f"v{len(self.registry['models']) + 1}"
        timestamp = datetime.datetime.utcnow().isoformat()

        model_path = self.model_dir / f"model_{version}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Compute a hash of the serialised model for integrity checking
        with open(model_path, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()[:12]

        entry = {
            "version": version,
            "timestamp": timestamp,
            "path": str(model_path),
            "hash": model_hash,
            "metadata": metadata,
        }
        self.registry["models"].append(entry)
        self.registry["active_version"] = version
        self._save_registry()
        logger.info("Registered model %s (hash=%s)", version, model_hash)
        return version

    def load_active(self):
        """Load the currently active model."""
        version = self.registry.get("active_version")
        if version is None:
            return None, None
        return self.load(version)

    def load(self, version: str):
        """Load a specific model version."""
        for entry in self.registry["models"]:
            if entry["version"] == version:
                with open(entry["path"], "rb") as f:
                    model = pickle.load(f)
                return model, entry
        return None, None

    def set_active(self, version: str) -> bool:
        versions = [m["version"] for m in self.registry["models"]]
        if version in versions:
            self.registry["active_version"] = version
            self._save_registry()
            return True
        return False

    def list_versions(self) -> list:
        return self.registry["models"]

    @property
    def active_version(self):
        return self.registry.get("active_version")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def train_model(algorithm: str = "random_forest", random_state: int = 42):
    """Train a model on the Iris dataset and return the pipeline + metrics."""
    data = load_iris()
    X = pd.DataFrame(data.data, columns=FEATURE_NAMES)
    y = pd.Series(data.target, name="species")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    if algorithm == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=100, random_state=random_state
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=100, random_state=random_state
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "classification_report": report,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    metadata = {
        "algorithm": algorithm,
        "random_state": random_state,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "metrics": metrics,
    }

    logger.info(
        "Trained %s – accuracy=%.4f, cv=%.4f +/- %.4f",
        algorithm, accuracy, cv_scores.mean(), cv_scores.std(),
    )
    return pipeline, metadata


# ---------------------------------------------------------------------------
# Request logging decorator
# ---------------------------------------------------------------------------

def log_request(f):
    """Decorator that logs every request and its response time."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())[:8]
        g.request_id = request_id
        start = time.time()

        logger.info(
            "[%s] %s %s | body=%s",
            request_id, request.method, request.path,
            request.get_data(as_text=True)[:500] if request.data else "-",
        )

        response = f(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000

        status = response[1] if isinstance(response, tuple) else 200
        logger.info("[%s] completed in %.1f ms | status=%s", request_id, elapsed_ms, status)
        return response

    return wrapper


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_features(data: dict) -> tuple:
    """Validate that input data contains the expected features.

    Returns (features_array, error_message).
    """
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return None, f"Missing features: {missing}"

    try:
        values = [float(data[f]) for f in FEATURE_NAMES]
    except (ValueError, TypeError) as exc:
        return None, f"Invalid feature value: {exc}"

    arr = np.array(values).reshape(1, -1)

    # Basic range check (Iris features are all positive and < 10)
    if np.any(arr < 0) or np.any(arr > 20):
        return None, "Feature values out of expected range (0-20)"

    return arr, None


def validate_batch(payload: list) -> tuple:
    """Validate a list of sample dicts for batch prediction."""
    if not isinstance(payload, list) or len(payload) == 0:
        return None, "Payload must be a non-empty list of sample objects"
    if len(payload) > 1000:
        return None, "Batch size exceeds maximum of 1000"

    arrays = []
    for idx, sample in enumerate(payload):
        arr, err = validate_features(sample)
        if err:
            return None, f"Sample {idx}: {err}"
        arrays.append(arr)

    return np.vstack(arrays), None


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def create_app(train_on_start: bool = False, algorithm: str = "random_forest"):
    """Application factory – creates and configures the Flask app."""
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    registry = ModelRegistry(MODEL_DIR)

    # Make sure we have at least one model
    model, model_entry = registry.load_active()
    if model is None or train_on_start:
        logger.info("Training initial model (%s) ...", algorithm)
        pipeline, metadata = train_model(algorithm=algorithm)
        registry.register(pipeline, metadata)
        model, model_entry = registry.load_active()

    # Store in app context so handlers can access them
    app.config["registry"] = registry
    app.config["model"] = model
    app.config["model_entry"] = model_entry

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    @log_request
    def health():
        """Health-check endpoint."""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "model_version": registry.active_version,
        })

    @app.route("/model-info", methods=["GET"])
    @log_request
    def model_info():
        """Return metadata for the active model."""
        entry = app.config["model_entry"]
        return jsonify({
            "version": entry["version"],
            "hash": entry["hash"],
            "timestamp": entry["timestamp"],
            "metadata": entry["metadata"],
        })

    @app.route("/model-versions", methods=["GET"])
    @log_request
    def model_versions():
        """List all registered model versions."""
        versions = registry.list_versions()
        return jsonify({
            "active_version": registry.active_version,
            "versions": [
                {
                    "version": v["version"],
                    "timestamp": v["timestamp"],
                    "hash": v["hash"],
                    "accuracy": v["metadata"].get("metrics", {}).get("accuracy"),
                }
                for v in versions
            ],
        })

    @app.route("/model-versions/<version>/activate", methods=["POST"])
    @log_request
    def activate_version(version):
        """Set a specific model version as active."""
        if registry.set_active(version):
            m, entry = registry.load(version)
            app.config["model"] = m
            app.config["model_entry"] = entry
            return jsonify({"message": f"Activated {version}"})
        return jsonify({"error": f"Version {version} not found"}), 404

    @app.route("/predict", methods=["POST"])
    @log_request
    def predict():
        """Single-sample prediction endpoint.

        Expects JSON:
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        """
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON payload"}), 400

        features, err = validate_features(data)
        if err:
            return jsonify({"error": err}), 422

        mdl = app.config["model"]
        prediction = int(mdl.predict(features)[0])
        probabilities = mdl.predict_proba(features)[0].tolist()

        return jsonify({
            "prediction": prediction,
            "predicted_class": TARGET_NAMES[prediction],
            "probabilities": {
                name: round(prob, 4) for name, prob in zip(TARGET_NAMES, probabilities)
            },
            "model_version": registry.active_version,
        })

    @app.route("/predict/batch", methods=["POST"])
    @log_request
    def predict_batch():
        """Batch prediction endpoint.

        Expects JSON:
        {
            "samples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                ...
            ]
        }
        """
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json(silent=True)
        if data is None or "samples" not in data:
            return jsonify({"error": "Payload must contain a 'samples' key"}), 400

        features, err = validate_batch(data["samples"])
        if err:
            return jsonify({"error": err}), 422

        mdl = app.config["model"]
        predictions = mdl.predict(features).tolist()
        probabilities = mdl.predict_proba(features).tolist()

        results = []
        for pred, probs in zip(predictions, probabilities):
            pred = int(pred)
            results.append({
                "prediction": pred,
                "predicted_class": TARGET_NAMES[pred],
                "probabilities": {
                    name: round(p, 4) for name, p in zip(TARGET_NAMES, probs)
                },
            })

        return jsonify({
            "results": results,
            "count": len(results),
            "model_version": registry.active_version,
        })

    @app.route("/train", methods=["POST"])
    @log_request
    def retrain():
        """Trigger model retraining.

        Optional JSON body:
        {"algorithm": "gradient_boosting", "random_state": 42}
        """
        data = request.get_json(silent=True) or {}
        algo = data.get("algorithm", "random_forest")
        seed = data.get("random_state", 42)

        pipeline, metadata = train_model(algorithm=algo, random_state=seed)
        version = registry.register(pipeline, metadata)

        # Hot-swap the active model
        app.config["model"] = pipeline
        _, entry = registry.load(version)
        app.config["model_entry"] = entry

        return jsonify({
            "message": f"Model retrained and registered as {version}",
            "version": version,
            "metrics": metadata["metrics"],
        })

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def internal_error(e):
        logger.exception("Internal server error")
        return jsonify({"error": "Internal server error"}), 500

    return app


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ML Model Serving API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=5000, help="Bind port")
    parser.add_argument("--train", action="store_true", help="Retrain model on startup")
    parser.add_argument(
        "--algorithm",
        default="random_forest",
        choices=["random_forest", "gradient_boosting"],
        help="Algorithm for initial training",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    app = create_app(train_on_start=args.train, algorithm=args.algorithm)
    logger.info("Starting server on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
