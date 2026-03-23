"""
Automated ML Pipeline with Experiment Tracking
===============================================
End-to-end machine learning pipeline that automates:
  1. Data ingestion and validation
  2. Preprocessing (imputation, scaling, encoding)
  3. Hyperparameter search
  4. Model training and evaluation
  5. Model selection and registration

Features:
  - Pipeline orchestration with step-level timing and status tracking
  - Experiment tracking: parameters, metrics, and artifacts logged to JSON
  - Model registry: save, load, compare, and promote models
  - Reproducibility: random seed propagation and data-version hashing
  - YAML-style dict configuration for the entire pipeline

Usage:
    python ml_pipeline.py                 # Run the default pipeline
    python ml_pipeline.py --config custom # Not implemented; extend as needed
"""

import os
import json
import time
import pickle
import hashlib
import datetime
import warnings
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
REGISTRY_DIR = BASE_DIR / "model_registry"

for _d in (ARTIFACTS_DIR, EXPERIMENTS_DIR, REGISTRY_DIR):
    _d.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data-version hashing
# ---------------------------------------------------------------------------

def hash_dataframe(df: pd.DataFrame) -> str:
    """Return a short SHA-256 hex digest of a DataFrame's content."""
    raw = pd.util.hash_pandas_object(df).values.tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Default pipeline configuration (YAML-style dict)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "pipeline_name": "iris_classification",
    "random_seed": 42,
    "data": {
        "source": "sklearn",           # sklearn | csv
        "dataset": "iris",             # iris | wine | breast_cancer
        "test_size": 0.2,
        "validation_size": 0.1,
    },
    "preprocessing": {
        "impute_strategy": "mean",     # mean | median | most_frequent
        "scale": True,
    },
    "training": {
        "algorithms": {
            "random_forest": {
                "enabled": True,
                "params": {"n_estimators": 100, "max_depth": None},
                "search_space": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                },
            },
            "gradient_boosting": {
                "enabled": True,
                "params": {"n_estimators": 100, "learning_rate": 0.1},
                "search_space": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                },
            },
            "logistic_regression": {
                "enabled": True,
                "params": {"max_iter": 1000},
                "search_space": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["lbfgs", "liblinear"],
                },
            },
        },
        "hyperparameter_search": {
            "method": "grid",       # grid | random
            "cv_folds": 5,
            "scoring": "accuracy",
            "n_iter": 20,           # only for random search
        },
    },
    "evaluation": {
        "primary_metric": "accuracy",
        "metrics": ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    },
    "model_selection": {
        "strategy": "best_metric",     # best_metric | within_tolerance
        "tolerance": 0.005,            # used only with within_tolerance
    },
}


# ---------------------------------------------------------------------------
# Experiment tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Logs parameters, metrics, and artifacts for each experiment run."""

    def __init__(self, experiment_name: str, base_dir: Path = EXPERIMENTS_DIR):
        self.experiment_name = experiment_name
        self.run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_dir = base_dir / experiment_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log: Dict[str, Any] = {
            "experiment": experiment_name,
            "run_id": self.run_id,
            "start_time": datetime.datetime.utcnow().isoformat(),
            "params": {},
            "metrics": {},
            "artifacts": [],
            "steps": [],
        }

    def log_params(self, params: dict):
        self.log["params"].update(_make_serialisable(params))

    def log_metric(self, key: str, value: float):
        self.log["metrics"][key] = round(value, 6)

    def log_metrics(self, metrics: dict):
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_artifact(self, name: str, obj: Any):
        path = self.run_dir / name
        if isinstance(obj, (dict, list)):
            with open(path.with_suffix(".json"), "w") as f:
                json.dump(obj, f, indent=2, default=str)
            self.log["artifacts"].append(str(path.with_suffix(".json")))
        else:
            with open(path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(obj, f)
            self.log["artifacts"].append(str(path.with_suffix(".pkl")))

    def log_step(self, step_name: str, status: str, duration_s: float, details: dict = None):
        entry = {
            "step": step_name,
            "status": status,
            "duration_seconds": round(duration_s, 3),
        }
        if details:
            entry["details"] = _make_serialisable(details)
        self.log["steps"].append(entry)

    def save(self):
        self.log["end_time"] = datetime.datetime.utcnow().isoformat()
        with open(self.run_dir / "run_log.json", "w") as f:
            json.dump(self.log, f, indent=2, default=str)
        print(f"  [ExperimentTracker] Run log saved to {self.run_dir / 'run_log.json'}")

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "metrics": self.log["metrics"],
            "steps": [(s["step"], s["status"]) for s in self.log["steps"]],
        }


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Persistent, file-based model registry for comparing and promoting models."""

    def __init__(self, base_dir: Path = REGISTRY_DIR):
        self.base_dir = base_dir
        self.index_path = base_dir / "index.json"
        self._load()

    def _load(self):
        if self.index_path.exists():
            with open(self.index_path) as f:
                self.index = json.load(f)
        else:
            self.index = {"models": [], "production_model": None}

    def _save(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2, default=str)

    def register(self, model, name: str, metrics: dict, params: dict, tags: dict = None):
        """Persist a model and add it to the registry."""
        version = sum(1 for m in self.index["models"] if m["name"] == name) + 1
        model_id = f"{name}_v{version}"
        model_path = self.base_dir / f"{model_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        entry = {
            "model_id": model_id,
            "name": name,
            "version": version,
            "path": str(model_path),
            "metrics": metrics,
            "params": _make_serialisable(params),
            "tags": tags or {},
            "registered_at": datetime.datetime.utcnow().isoformat(),
        }
        self.index["models"].append(entry)
        self._save()
        print(f"  [Registry] Registered {model_id}")
        return model_id

    def load_model(self, model_id: str):
        for entry in self.index["models"]:
            if entry["model_id"] == model_id:
                with open(entry["path"], "rb") as f:
                    return pickle.load(f)
        raise ValueError(f"Model {model_id} not found in registry")

    def compare(self, metric: str = "accuracy") -> pd.DataFrame:
        """Return a DataFrame comparing all registered models on the given metric."""
        rows = []
        for m in self.index["models"]:
            rows.append({
                "model_id": m["model_id"],
                "name": m["name"],
                "version": m["version"],
                metric: m["metrics"].get(metric, float("nan")),
                "registered_at": m["registered_at"],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(metric, ascending=False).reset_index(drop=True)
        return df

    def promote_to_production(self, model_id: str):
        ids = [m["model_id"] for m in self.index["models"]]
        if model_id not in ids:
            raise ValueError(f"{model_id} not found")
        self.index["production_model"] = model_id
        self._save()
        print(f"  [Registry] Promoted {model_id} to production")

    @property
    def production_model_id(self):
        return self.index.get("production_model")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}

ALGORITHM_MAP = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "adaboost": AdaBoostClassifier,
    "svm": SVC,
}


def step_data_ingestion(config: dict, tracker: ExperimentTracker) -> dict:
    """Load data from a configured source."""
    t0 = time.time()
    data_cfg = config["data"]
    seed = config["random_seed"]

    if data_cfg["source"] == "sklearn":
        loader = DATASET_LOADERS[data_cfg["dataset"]]
        raw = loader()
        X = pd.DataFrame(raw.data, columns=[f"f{i}" for i in range(raw.data.shape[1])])
        y = pd.Series(raw.target, name="target")
        target_names = list(raw.target_names) if hasattr(raw, "target_names") else None
    elif data_cfg["source"] == "csv":
        df = pd.read_csv(data_cfg["path"])
        target_col = data_cfg.get("target_column", "target")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        target_names = None
    else:
        raise ValueError(f"Unknown data source: {data_cfg['source']}")

    data_hash = hash_dataframe(pd.concat([X, y], axis=1))
    tracker.log_params({"data_hash": data_hash, "n_samples": len(X), "n_features": X.shape[1]})

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=seed, stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=data_cfg.get("validation_size", 0.1),
        random_state=seed,
        stratify=y_train,
    )

    result = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "target_names": target_names, "data_hash": data_hash,
    }

    duration = time.time() - t0
    tracker.log_step("data_ingestion", "success", duration, {
        "train": len(X_train), "val": len(X_val), "test": len(X_test),
    })
    print(f"  [1/5] Data ingestion complete – train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return result


def step_preprocessing(data: dict, config: dict, tracker: ExperimentTracker) -> dict:
    """Impute missing values and optionally scale features."""
    t0 = time.time()
    pre_cfg = config["preprocessing"]

    imputer = SimpleImputer(strategy=pre_cfg["impute_strategy"])
    scaler = StandardScaler() if pre_cfg["scale"] else None

    X_train = pd.DataFrame(
        imputer.fit_transform(data["X_train"]),
        columns=data["X_train"].columns,
        index=data["X_train"].index,
    )
    X_val = pd.DataFrame(
        imputer.transform(data["X_val"]),
        columns=data["X_val"].columns,
        index=data["X_val"].index,
    )
    X_test = pd.DataFrame(
        imputer.transform(data["X_test"]),
        columns=data["X_test"].columns,
        index=data["X_test"].index,
    )

    if scaler:
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index,
        )
        X_val = pd.DataFrame(
            scaler.transform(X_val), columns=X_val.columns, index=X_val.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index,
        )

    data["X_train"] = X_train
    data["X_val"] = X_val
    data["X_test"] = X_test
    data["imputer"] = imputer
    data["scaler"] = scaler

    tracker.log_params({"impute_strategy": pre_cfg["impute_strategy"], "scaled": pre_cfg["scale"]})

    duration = time.time() - t0
    tracker.log_step("preprocessing", "success", duration)
    print(f"  [2/5] Preprocessing complete – impute={pre_cfg['impute_strategy']}, scale={pre_cfg['scale']}")
    return data


def step_hyperparameter_search(data: dict, config: dict, tracker: ExperimentTracker) -> dict:
    """Run hyperparameter search for every enabled algorithm."""
    t0 = time.time()
    training_cfg = config["training"]
    search_cfg = training_cfg["hyperparameter_search"]
    seed = config["random_seed"]

    best_models = {}

    for algo_name, algo_cfg in training_cfg["algorithms"].items():
        if not algo_cfg.get("enabled", False):
            continue

        print(f"    -> Searching {algo_name} ...")
        cls = ALGORITHM_MAP[algo_name]

        base_params = deepcopy(algo_cfg["params"])
        if "random_state" in cls().get_params():
            base_params["random_state"] = seed

        estimator = cls(**base_params)
        search_space = algo_cfg.get("search_space", {})

        if not search_space:
            # No search space: just use defaults
            estimator.fit(data["X_train"], data["y_train"])
            best_models[algo_name] = {
                "estimator": estimator,
                "best_params": base_params,
                "best_score": cross_val_score(
                    estimator, data["X_train"], data["y_train"],
                    cv=search_cfg["cv_folds"], scoring=search_cfg["scoring"],
                ).mean(),
            }
            continue

        if search_cfg["method"] == "grid":
            searcher = GridSearchCV(
                estimator, search_space,
                cv=search_cfg["cv_folds"],
                scoring=search_cfg["scoring"],
                n_jobs=-1,
                refit=True,
            )
        else:
            searcher = RandomizedSearchCV(
                estimator, search_space,
                n_iter=search_cfg["n_iter"],
                cv=search_cfg["cv_folds"],
                scoring=search_cfg["scoring"],
                n_jobs=-1,
                refit=True,
                random_state=seed,
            )

        searcher.fit(data["X_train"], data["y_train"])

        best_models[algo_name] = {
            "estimator": searcher.best_estimator_,
            "best_params": searcher.best_params_,
            "best_score": searcher.best_score_,
        }
        tracker.log_params({f"{algo_name}_best_params": searcher.best_params_})
        tracker.log_metric(f"{algo_name}_cv_score", searcher.best_score_)
        print(f"      best_score={searcher.best_score_:.4f}  params={searcher.best_params_}")

    data["best_models"] = best_models

    duration = time.time() - t0
    tracker.log_step("hyperparameter_search", "success", duration, {
        "algorithms_searched": list(best_models.keys()),
    })
    print(f"  [3/5] Hyperparameter search complete – {len(best_models)} algorithms evaluated")
    return data


def step_evaluation(data: dict, config: dict, tracker: ExperimentTracker) -> dict:
    """Evaluate all candidate models on the validation set."""
    t0 = time.time()
    eval_cfg = config["evaluation"]
    results = {}

    for algo_name, info in data["best_models"].items():
        model = info["estimator"]
        y_pred = model.predict(data["X_val"])

        metrics = {
            "accuracy": accuracy_score(data["y_val"], y_pred),
            "precision_macro": precision_score(data["y_val"], y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(data["y_val"], y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(data["y_val"], y_pred, average="macro", zero_division=0),
            "cv_score": info["best_score"],
        }
        results[algo_name] = metrics
        tracker.log_metrics({f"{algo_name}_{k}": v for k, v in metrics.items()})
        print(f"    {algo_name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

    data["evaluation_results"] = results
    tracker.log_artifact("evaluation_results", results)

    duration = time.time() - t0
    tracker.log_step("evaluation", "success", duration)
    print(f"  [4/5] Evaluation complete")
    return data


def step_model_selection(
    data: dict,
    config: dict,
    tracker: ExperimentTracker,
    registry: ModelRegistry,
) -> dict:
    """Select the best model, register it, and promote to production."""
    t0 = time.time()
    sel_cfg = config["model_selection"]
    primary = config["evaluation"]["primary_metric"]

    # Rank candidates
    ranked = sorted(
        data["evaluation_results"].items(),
        key=lambda kv: kv[1][primary],
        reverse=True,
    )

    best_name, best_metrics = ranked[0]
    best_model = data["best_models"][best_name]["estimator"]
    best_params = data["best_models"][best_name]["best_params"]

    # Final evaluation on held-out test set
    y_test_pred = best_model.predict(data["X_test"])
    test_metrics = {
        "test_accuracy": accuracy_score(data["y_test"], y_test_pred),
        "test_f1_macro": f1_score(data["y_test"], y_test_pred, average="macro", zero_division=0),
    }
    tracker.log_metrics(test_metrics)

    print(f"  [5/5] Selected model: {best_name}")
    print(f"         Validation {primary}: {best_metrics[primary]:.4f}")
    print(f"         Test accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"         Test F1:       {test_metrics['test_f1_macro']:.4f}")

    # Register and promote
    all_metrics = {**best_metrics, **test_metrics}
    model_id = registry.register(
        best_model,
        name=best_name,
        metrics=all_metrics,
        params=best_params,
        tags={"pipeline": config["pipeline_name"], "data_hash": data.get("data_hash", "")},
    )
    registry.promote_to_production(model_id)

    tracker.log_params({"selected_model": best_name, "model_id": model_id})
    tracker.log_artifact("best_model", best_model)

    data["selected_model_id"] = model_id
    data["selected_model_name"] = best_name
    data["test_metrics"] = test_metrics

    duration = time.time() - t0
    tracker.log_step("model_selection", "success", duration)
    return data


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class MLPipeline:
    """Orchestrates the full ML pipeline from ingestion to deployment."""

    def __init__(self, config: dict = None):
        self.config = config or deepcopy(DEFAULT_CONFIG)
        self.tracker = ExperimentTracker(self.config["pipeline_name"])
        self.registry = ModelRegistry()
        np.random.seed(self.config["random_seed"])

    def run(self) -> dict:
        """Execute every pipeline step in sequence and return a summary."""
        print(f"\n{'='*60}")
        print(f"  ML Pipeline: {self.config['pipeline_name']}")
        print(f"  Run ID:      {self.tracker.run_id}")
        print(f"  Seed:        {self.config['random_seed']}")
        print(f"{'='*60}\n")

        self.tracker.log_params(self.config)
        t_total = time.time()

        # Step 1 – Data Ingestion
        data = step_data_ingestion(self.config, self.tracker)

        # Step 2 – Preprocessing
        data = step_preprocessing(data, self.config, self.tracker)

        # Step 3 – Hyperparameter Search / Training
        data = step_hyperparameter_search(data, self.config, self.tracker)

        # Step 4 – Evaluation
        data = step_evaluation(data, self.config, self.tracker)

        # Step 5 – Model Selection & Registration
        data = step_model_selection(data, self.config, self.tracker, self.registry)

        total_duration = time.time() - t_total
        self.tracker.log_metric("total_pipeline_seconds", total_duration)
        self.tracker.save()

        # Summary
        print(f"\n{'='*60}")
        print(f"  Pipeline finished in {total_duration:.1f}s")
        print(f"  Production model: {self.registry.production_model_id}")
        print(f"{'='*60}")

        # Show registry comparison
        comparison = self.registry.compare()
        if not comparison.empty:
            print("\n  Model Registry:\n")
            print(comparison.to_string(index=False))

        return {
            "run_id": self.tracker.run_id,
            "selected_model": data["selected_model_name"],
            "model_id": data["selected_model_id"],
            "test_metrics": data["test_metrics"],
            "total_seconds": round(total_duration, 2),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_serialisable(obj):
    """Recursively convert numpy/pandas types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Run 1 – default (Iris)
    pipeline = MLPipeline()
    summary = pipeline.run()
    print(f"\n  Summary: {json.dumps(summary, indent=2)}\n")

    # Run 2 – Wine dataset with different configuration
    wine_config = deepcopy(DEFAULT_CONFIG)
    wine_config["pipeline_name"] = "wine_classification"
    wine_config["data"]["dataset"] = "wine"
    wine_config["random_seed"] = 123

    pipeline2 = MLPipeline(config=wine_config)
    summary2 = pipeline2.run()
    print(f"\n  Summary: {json.dumps(summary2, indent=2)}\n")


if __name__ == "__main__":
    main()
