# logger.py
import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, Optional


class MetricLogger:
    def __init__(self, base_log_dir: str = "logs", run_name: Optional[str] = None):
        """
        Creates a new run folder like logs/run_2025-12-06_145500
        and sets up metrics.csv + hparams.json paths.
        """
        os.makedirs(base_log_dir, exist_ok=True)

        if run_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"run_{timestamp}"

        self.run_name = run_name
        self.run_dir = os.path.join(base_log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self.hparams_path = os.path.join(self.run_dir, "hparams.json")
        self.videos_dir = os.path.join(self.run_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

        self._metrics_file = None
        self._csv_writer = None
        self._fieldnames = None
        self._buffer = []  # optional, if you want buffering

    def save_hparams(self, hparams: Dict[str, Any]):
        """Save hyperparameters / config to hparams.json."""
        with open(self.hparams_path, "w") as f:
            json.dump(hparams, f, indent=4)

    def _init_csv(self, metrics: Dict[str, Any]):
        """Initialize metrics.csv and CSV writer if needed."""
        if self._csv_writer is not None:
            return

        # Fix column order: step first if provided
        fieldnames = list(metrics.keys())
        if "step" in fieldnames:
            fieldnames.insert(0, fieldnames.pop(fieldnames.index("step")))

        self._fieldnames = fieldnames
        file_exists = os.path.exists(self.metrics_path)

        self._metrics_file = open(self.metrics_path, mode="a", newline="")
        self._csv_writer = csv.DictWriter(
            self._metrics_file, fieldnames=self._fieldnames)

        if not file_exists:
            self._csv_writer.writeheader()

    def log(self, **metrics):
        """
        Log a single metrics row.
        Example:
            logger.log(step=total_steps, avg_reward=avg_reward, entropy=entropy)
        """
        if not metrics:
            return

        # Ensure everything is JSON-serializable-ish (e.g., floats, ints)
        clean_metrics = {}
        for k, v in metrics.items():
            try:
                float(v)
                clean_metrics[k] = v
            except Exception:
                # fallback: string
                clean_metrics[k] = str(v)

        if self._csv_writer is None:
            self._init_csv(clean_metrics)

        # Fill missing columns with empty string
        for name in self._fieldnames:
            if name not in clean_metrics:
                clean_metrics[name] = ""

        self._csv_writer.writerow(clean_metrics)
        self._metrics_file.flush()

    def close(self):
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
            self._csv_writer = None
