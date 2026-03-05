"""Train all anomaly detection models and save evaluation results.

Usage:
    python -m scripts.train_and_evaluate [--data-path DATA_CSV]

Generates ``data/evaluation_results.json`` consumed by the dashboard's
Model Comparison tab.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.generator import IoTDataGenerator
from src.data.preprocessor import DataPreprocessor
from src.models.autoencoder import AutoencoderDetector
from src.models.dbscan_detector import DBSCANDetector
from src.models.evaluator import ModelEvaluator
from src.models.isolation_forest import IsolationForestDetector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DATA_DIR = Path("data")
GENERATED_CSV = DATA_DIR / "generated" / "sensor_data.csv"
EVAL_OUTPUT = DATA_DIR / "evaluation_results.json"


def load_or_generate_data(data_path: Path) -> pd.DataFrame:
    if data_path.exists():
        logger.info("Loading existing data from %s", data_path)
        return pd.read_csv(data_path, parse_dates=["timestamp"])
    logger.info("No data found — generating 30 days of synthetic data")
    gen = IoTDataGenerator()
    df = gen.generate(days=30)
    gen.save_to_csv(df, str(data_path))
    return df


def main(data_path: Path = GENERATED_CSV) -> None:
    df = load_or_generate_data(data_path)

    # Preprocess
    preprocessor = DataPreprocessor(scaler_type="standard")
    pivoted = preprocessor.pivot_sensors(df)
    X_scaled, y = preprocessor.fit_transform(pivoted)
    feature_names = preprocessor.feature_names

    if y is None:
        raise RuntimeError("Dataset has no is_anomaly labels — cannot evaluate")

    # Train/test split (temporal)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(
        X_scaled, y, train_ratio=0.7, val_ratio=0.15
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test)
    )

    results: dict[str, dict] = {}

    # --- Isolation Forest ---
    logger.info("Training Isolation Forest...")
    iso = IsolationForestDetector(contamination=0.05)
    iso.fit(X_train, feature_names)
    iso_pred = iso.predict(X_test)
    iso_scores = iso.score_samples(X_test)
    iso_eval = ModelEvaluator.evaluate(y_test, iso_pred, iso_scores)
    results["Isolation Forest"] = iso_eval.to_dict()
    iso.save(str(DATA_DIR / "models" / "isolation_forest.joblib"))

    # --- Autoencoder ---
    logger.info("Training Autoencoder...")
    ae = AutoencoderDetector(epochs=50, lr=0.001)
    ae.fit(X_train, feature_names, X_val=X_val)
    ae_pred = ae.predict(X_test)
    ae_scores = ae.score_samples(X_test)
    ae_eval = ModelEvaluator.evaluate(y_test, ae_pred, ae_scores)
    results["Autoencoder"] = ae_eval.to_dict()
    ae.save(str(DATA_DIR / "models" / "autoencoder.pt"))

    # --- DBSCAN ---
    logger.info("Training DBSCAN...")
    dbscan = DBSCANDetector()
    dbscan.fit(X_train, feature_names, auto_tune=True)
    dbscan_pred = dbscan.predict(X_test)
    dbscan_scores = dbscan.score_samples(X_test)
    dbscan_eval = ModelEvaluator.evaluate(y_test, dbscan_pred, dbscan_scores)
    results["DBSCAN"] = dbscan_eval.to_dict()
    dbscan.save(str(DATA_DIR / "models" / "dbscan.joblib"))

    # --- Print comparison ---
    from src.models.evaluator import EvaluationResult

    eval_objects = {}
    for name, d in results.items():
        eval_objects[name] = EvaluationResult(
            precision=d["precision"],
            recall=d["recall"],
            f1=d["f1"],
            auc_roc=d["auc_roc"],
            confusion_matrix=np.array(d["confusion_matrix"]),
        )
    print(ModelEvaluator.compare_models(eval_objects))

    # --- Save results for dashboard ---
    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation results to %s", EVAL_OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate anomaly detection models")
    parser.add_argument(
        "--data-path", type=Path, default=GENERATED_CSV, help="Path to sensor CSV data"
    )
    args = parser.parse_args()
    main(data_path=args.data_path)
