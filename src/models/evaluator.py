"""Model evaluation metrics and comparison reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics.

    Attributes:
        precision: Precision score.
        recall: Recall score.
        f1: F1 score.
        auc_roc: Area under the ROC curve (0 if scores unavailable).
        confusion_matrix: 2x2 confusion matrix.
    """

    precision: float
    recall: float
    f1: float
    auc_roc: float
    confusion_matrix: np.ndarray

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


class ModelEvaluator:
    """Compute evaluation metrics and generate comparison reports."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray | None = None,
    ) -> EvaluationResult:
        """Compute classification metrics.

        Args:
            y_true: Ground-truth binary labels.
            y_pred: Predicted binary labels.
            y_scores: Continuous anomaly scores (for AUC-ROC).

        Returns:
            :class:`EvaluationResult` with computed metrics.
        """
        auc = 0.0
        if y_scores is not None:
            try:
                auc = float(roc_auc_score(y_true, y_scores))
            except ValueError:
                auc = 0.0

        return EvaluationResult(
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            auc_roc=auc,
            confusion_matrix=confusion_matrix(y_true, y_pred),
        )

    @staticmethod
    def compare_models(results: dict[str, EvaluationResult]) -> str:
        """Generate a plaintext comparison report.

        Args:
            results: Mapping of model name to evaluation result.

        Returns:
            Formatted report string.
        """
        report = "Model Comparison Report\n" + "=" * 50 + "\n\n"
        report += f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC-ROC':<12}\n"
        report += "-" * 68 + "\n"

        for name, result in results.items():
            report += (
                f"{name:<20} {result.precision:<12.4f} {result.recall:<12.4f} "
                f"{result.f1:<12.4f} {result.auc_roc:<12.4f}\n"
            )

        best_model = max(results.items(), key=lambda x: x[1].f1)
        report += "\n" + "=" * 50 + "\n"
        report += f"Best Model (by F1): {best_model[0]} (F1={best_model[1].f1:.4f})\n"
        return report
