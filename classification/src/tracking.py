"""
MLflow experiment tracking utilities.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


def start_run(
    experiment_name: str,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
):
    """Start an MLflow run. Returns the active run."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_params(params: Dict[str, Any]):
    """Log dictionary of parameters to MLflow."""
    for key, value in params.items():
        try:
            mlflow.log_param(key, value)
        except Exception:
            pass


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Log dictionary of metrics to MLflow."""
    for key, value in metrics.items():
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception:
            pass


def log_artifact(path: Path):
    """Log a file artifact to MLflow."""
    if path.exists():
        mlflow.log_artifact(str(path))


def log_model_summary(summary: Dict[str, Any], path: Path):
    """Save and log pipeline summary JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    log_artifact(path)
