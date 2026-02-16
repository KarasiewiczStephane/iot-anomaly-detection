"""DBSCAN-based anomaly detector where noise points are classified as anomalies."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DBSCANDetector:
    """Density-based anomaly detector using DBSCAN.

    Points that fall outside any cluster (noise) are treated as
    anomalies. For new unseen data, the average k-nearest-neighbour
    distance to the training set determines anomaly status.

    Args:
        eps: Maximum distance between two samples in the same neighbourhood.
        min_samples: Minimum cluster size.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.model: DBSCAN | None = None
        self.training_data: np.ndarray | None = None
        self.feature_names: list[str] = []

    @staticmethod
    def find_optimal_eps(X: np.ndarray, min_samples: int = 5) -> float:
        """Estimate a good *eps* via the k-distance elbow method.

        Args:
            X: Feature array.
            min_samples: Number of nearest neighbours to consider.

        Returns:
            Suggested epsilon value.
        """
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        k_distances = np.sort(distances[:, -1])

        n = len(k_distances)
        coords = np.column_stack((np.arange(n), k_distances))
        line_vec = coords[-1] - coords[0]
        line_vec_norm = line_vec / (np.linalg.norm(line_vec) + 1e-10)

        vec_from_first = coords - coords[0]
        scalar_proj = np.dot(vec_from_first, line_vec_norm)
        vec_to_line = vec_from_first - np.outer(scalar_proj, line_vec_norm)
        dist_to_line = np.linalg.norm(vec_to_line, axis=1)

        elbow_idx = int(np.argmax(dist_to_line))
        return float(k_distances[elbow_idx])

    def fit(
        self,
        X: np.ndarray,
        feature_names: list[str],
        auto_tune: bool = False,
    ) -> DBSCANDetector:
        """Fit the DBSCAN model on training data.

        Args:
            X: Training feature array.
            feature_names: Column names.
            auto_tune: If ``True``, estimate *eps* from data before fitting.

        Returns:
            ``self`` for method chaining.
        """
        self.feature_names = feature_names
        self.training_data = X.copy()

        if auto_tune:
            self.eps = self.find_optimal_eps(X, self.min_samples)
            logger.info("Auto-tuned eps=%.4f", self.eps)

        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        self.model.fit(X)
        logger.info(
            "DBSCAN fitted: %d clusters, %d noise points",
            len(set(self.model.labels_) - {-1}),
            (self.model.labels_ == -1).sum(),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies for new data.

        Points whose average distance to the *k* nearest training
        neighbours exceeds *eps* are flagged as anomalies.

        Args:
            X: Feature array.

        Returns:
            Binary prediction array (1 = anomaly).
        """
        neighbors = NearestNeighbors(n_neighbors=self.min_samples)
        neighbors.fit(self.training_data)
        distances, _ = neighbors.kneighbors(X)
        avg_distances = distances.mean(axis=1)
        return (avg_distances > self.eps).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Normalised anomaly scores in [0, 1] based on neighbour distance.

        Args:
            X: Feature array.

        Returns:
            Score array.
        """
        neighbors = NearestNeighbors(n_neighbors=self.min_samples)
        neighbors.fit(self.training_data)
        distances, _ = neighbors.kneighbors(X)
        avg_distances = distances.mean(axis=1)
        dist_range = avg_distances.max() - avg_distances.min() + 1e-10
        return (avg_distances - avg_distances.min()) / dist_range

    def get_feature_contributions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Per-feature distance contributions for root-cause analysis.

        Args:
            X: Feature array.

        Returns:
            Dict mapping feature name to normalised distance array.
        """
        contributions: dict[str, np.ndarray] = {}
        for i, name in enumerate(self.feature_names):
            neighbors = NearestNeighbors(n_neighbors=self.min_samples)
            neighbors.fit(self.training_data[:, i : i + 1])
            distances, _ = neighbors.kneighbors(X[:, i : i + 1])
            avg_dist = distances.mean(axis=1)
            dist_range = avg_dist.max() - avg_dist.min() + 1e-10
            contributions[name] = (avg_dist - avg_dist.min()) / dist_range
        return contributions

    def save(self, path: str) -> None:
        """Persist model to disk.

        Args:
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "training_data": self.training_data,
                "feature_names": self.feature_names,
                "config": {"eps": self.eps, "min_samples": self.min_samples},
            },
            path,
        )
        logger.info("Saved DBSCAN model to %s", path)

    @classmethod
    def load(cls, path: str) -> DBSCANDetector:
        """Load model from disk.

        Args:
            path: File path saved by :meth:`save`.

        Returns:
            Restored detector instance.
        """
        data = joblib.load(path)
        detector = cls(eps=data["config"]["eps"], min_samples=data["config"]["min_samples"])
        detector.model = data["model"]
        detector.training_data = data["training_data"]
        detector.feature_names = data["feature_names"]
        return detector
