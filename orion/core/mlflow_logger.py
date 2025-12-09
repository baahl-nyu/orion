import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowLogger:
    """
    MLflow integration for logging Orion FHE framework metrics, parameters,
    traces, and artifacts.

    Logs:
    - Model metrics: MAE, precision, inference time
    - System metrics: CPU/memory usage, compile time, fit time
    - Parameters: FHE scheme configuration, network architecture
    - Artifacts: Model weights, config files, DAG visualizations
    """

    def __init__(
        self,
        enabled: bool = True,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow logger.

        Args:
            enabled: Whether to enable MLflow logging
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (e.g., 'http://localhost:5000' or 'mlruns')
        """
        self.enabled = enabled and MLFLOW_AVAILABLE

        if self.enabled:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            # Enable automatic system metrics logging
            mlflow.enable_system_metrics_logging()

            self._timers = {}

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        """Start a new MLflow run."""
        if self.enabled:
            mlflow.start_run(run_name=run_name)
            if tags:
                mlflow.set_tags(tags)

    def end_run(self):
        """End the current MLflow run."""
        if self.enabled:
            mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if self.enabled:
            mlflow.log_params(params)

    def log_param(self, key: str, value: Any):
        """Log a single parameter to MLflow."""
        if self.enabled:
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if self.enabled:
            mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric to MLflow."""
        if self.enabled:
            mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to MLflow."""
        if self.enabled:
            import os
            if not os.path.exists(local_path):
                print(f"Warning: Artifact file not found: {local_path}")
                return
            try:
                print(f"Logging artifact: {local_path} to {artifact_path or 'root'}")
                mlflow.log_artifact(local_path, artifact_path)
                print("  ✓ Successfully logged artifact")
            except Exception as e:
                print(f"  ✗ Failed to log artifact: {e}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """Log a dictionary as a JSON artifact."""
        if self.enabled:
            try:
                print(f"Logging dict artifact: {artifact_file}")
                mlflow.log_dict(dictionary, artifact_file)
                print("  ✓ Successfully logged dict artifact")
            except Exception as e:
                print(f"  ✗ Failed to log dict artifact: {e}")


    @contextmanager
    def timer(self, name: str):
        """
        Context manager for timing operations and logging them.

        Usage:
            with logger.timer("compile"):
                # ... code to time
        """
        start = time.time()
        yield
        duration = time.time() - start
        self.log_metric(f"04_performance/time/{name}_seconds", duration)

    def log_scheme_params(self, scheme):
        """Log FHE scheme parameters."""
        if not self.enabled:
            return

        params = scheme.params
        self.log_params(
            {
                "backend": params.get_backend(),
                "log_n": params.get_logn(),
                "log_q": str(params.get_logq()),
                "margin": params.get_margin(),
                "fuse_modules": params.get_fuse_modules(),
            }
        )

    def log_network_stats(self, net, network_dag=None):
        """Log network architecture statistics."""
        if not self.enabled:
            return

        total_params = sum(p.numel() for p in net.parameters())
        self.log_metric("02_architecture/network/total_parameters", float(total_params))

        if network_dag:
            num_nodes = len(network_dag.nodes)
            self.log_metric("02_architecture/network/num_layers", float(num_nodes))

    def log_layer_stats(self, module, prefix: str = ""):
        """Log per-layer statistics."""
        if not self.enabled:
            return

        prefix = f"02_architecture/layers/{prefix}" if prefix else "02_architecture/layers"
        prefix = f"{prefix}/" if not prefix.endswith("/") else prefix

        if hasattr(module, "input_min") and hasattr(module, "input_max"):
            input_min = module.input_min.detach() if hasattr(module.input_min, 'detach') else module.input_min
            input_max = module.input_max.detach() if hasattr(module.input_max, 'detach') else module.input_max
            self.log_metrics(
                {
                    f"{prefix}input_min": float(input_min),
                    f"{prefix}input_max": float(input_max),
                }
            )

        if hasattr(module, "output_min") and hasattr(module, "output_max"):
            output_min = module.output_min.detach() if hasattr(module.output_min, 'detach') else module.output_min
            output_max = module.output_max.detach() if hasattr(module.output_max, 'detach') else module.output_max
            self.log_metrics(
                {
                    f"{prefix}output_min": float(output_min),
                    f"{prefix}output_max": float(output_max),
                }
            )

        if hasattr(module, "level"):
            self.log_metric(f"{prefix}level", float(module.level))

    def log_bootstrap_info(self, num_bootstraps: int, input_level: int):
        """Log bootstrap placement information."""
        if self.enabled:
            self.log_metrics(
                {
                    "03_fhe/bootstrap/num_operations": float(num_bootstraps),
                    "03_fhe/bootstrap/input_level": float(input_level),
                }
            )

    def log_inference_metrics(self, mae: float, precision: float, runtime: float):
        """Log FHE inference metrics."""
        if self.enabled:
            self.log_metrics(
                {
                    "03_fhe/inference/mae": mae,
                    "03_fhe/inference/precision": precision,
                    "03_fhe/inference/runtime_seconds": runtime,
                }
            )

    def log_model(self, model, artifact_path: str = "model"):
        """
        Log a PyTorch model to MLflow.

        Args:
            model: PyTorch model to log
            artifact_path: Path within the artifact URI to save the model
        """
        if self.enabled:
            try:
                print(f"Logging model to: {artifact_path}")
                mlflow.pytorch.log_model(model, artifact_path)
                print("  ✓ Successfully logged model")
            except Exception as e:
                print(f"  ✗ Failed to log model: {e}")


_global_logger: Optional[MLflowLogger] = None


def get_logger() -> MLflowLogger:
    """Get the global MLflow logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = MLflowLogger(enabled=False)
    return _global_logger


def set_logger(logger: MLflowLogger):
    """Set the global MLflow logger instance."""
    global _global_logger
    _global_logger = logger
