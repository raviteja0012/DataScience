# ML Model Serving API (Flask)

Production-grade REST API for serving scikit-learn models, with model versioning,
batch predictions, input validation, and request logging.

## Quick Start

```bash
pip install -r requirements.txt
python app.py              # start with default model
python app.py --train      # retrain before starting
python app.py --port 8080  # custom port
```

## API Reference

### Health Check

```
GET /health
```

Returns server health status and the active model version.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-01-15T12:00:00",
    "model_version": "v1"
}
```

### Model Info

```
GET /model-info
```

Returns metadata for the currently active model, including training metrics,
algorithm, and feature configuration.

### List Model Versions

```
GET /model-versions
```

Returns all registered model versions with their accuracy scores.

### Activate Model Version

```
POST /model-versions/<version>/activate
```

Switches the active model to a previously registered version.

### Single Prediction

```
POST /predict
Content-Type: application/json

{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

**Response:**
```json
{
    "prediction": 0,
    "predicted_class": "setosa",
    "probabilities": {
        "setosa": 0.97,
        "versicolor": 0.02,
        "virginica": 0.01
    },
    "model_version": "v1"
}
```

### Batch Prediction

```
POST /predict/batch
Content-Type: application/json

{
    "samples": [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5}
    ]
}
```

**Response:**
```json
{
    "results": [
        {"prediction": 0, "predicted_class": "setosa", "probabilities": {"setosa": 0.97, "versicolor": 0.02, "virginica": 0.01}},
        {"prediction": 1, "predicted_class": "versicolor", "probabilities": {"setosa": 0.01, "versicolor": 0.92, "virginica": 0.07}}
    ],
    "count": 2,
    "model_version": "v1"
}
```

Maximum batch size: 1000 samples.

### Retrain Model

```
POST /train
Content-Type: application/json

{
    "algorithm": "gradient_boosting",
    "random_state": 42
}
```

Triggers a full retrain, registers the new model version, and hot-swaps it as
the active model.

## Error Handling

All errors return JSON with an `error` key:

| Status | Meaning |
|--------|---------|
| 400 | Bad request (invalid JSON or missing Content-Type) |
| 404 | Endpoint not found |
| 405 | HTTP method not allowed |
| 422 | Validation error (missing/invalid features) |
| 500 | Internal server error |

## Architecture

```
app.py
  +-- ModelRegistry      # file-based model versioning
  +-- train_model()      # training with cross-validation
  +-- validate_features() / validate_batch()  # input checks
  +-- create_app()       # Flask application factory
models/                  # serialised model files + registry.json
logs/                    # structured request logs
```
