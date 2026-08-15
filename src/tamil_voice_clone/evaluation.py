from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Initial clone-quality gates.

    These are engineering thresholds, not universal perceptual guarantees. They
    should be calibrated after the first several bilingual checkpoints.
    """

    min_global_similarity: float = 0.72
    min_window_similarity: float = 0.64
    max_window_similarity_spread: float = 0.16
    max_adjacent_window_drop: float = 0.12
    max_word_error_rate: float = 0.08
    require_pronunciation_review: bool = True


@dataclass(frozen=True)
class CloneMetrics:
    global_similarity: float
    window_similarities: tuple[float, ...]
    word_error_rate: float | None = None
    pronunciation_review_passed: bool | None = None
    no_garbling_review_passed: bool | None = None

    @property
    def minimum_window_similarity(self) -> float:
        return min(self.window_similarities) if self.window_similarities else self.global_similarity

    @property
    def window_similarity_spread(self) -> float:
        values = self.window_similarities
        return max(values) - min(values) if values else 0.0

    @property
    def maximum_adjacent_drop(self) -> float:
        values = self.window_similarities
        if len(values) < 2:
            return 0.0
        return max(abs(current - previous) for previous, current in pairwise(values))


@dataclass(frozen=True)
class AcceptanceResult:
    accepted: bool
    failures: tuple[str, ...]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Embedding shapes differ: {a.shape} != {b.shape}")

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-8 or not np.isfinite(denominator):
        raise ValueError("Cannot compare empty or invalid embeddings")
    return float(np.dot(a, b) / denominator)


def build_clone_metrics(
    reference_embedding: np.ndarray,
    generated_embedding: np.ndarray,
    generated_window_embeddings: list[np.ndarray],
    *,
    word_error_rate: float | None = None,
    pronunciation_review_passed: bool | None = None,
    no_garbling_review_passed: bool | None = None,
) -> CloneMetrics:
    return CloneMetrics(
        global_similarity=cosine_similarity(reference_embedding, generated_embedding),
        window_similarities=tuple(
            cosine_similarity(reference_embedding, window)
            for window in generated_window_embeddings
        ),
        word_error_rate=word_error_rate,
        pronunciation_review_passed=pronunciation_review_passed,
        no_garbling_review_passed=no_garbling_review_passed,
    )


def evaluate_clone(
    metrics: CloneMetrics,
    thresholds: AcceptanceThresholds | None = None,
) -> AcceptanceResult:
    limits = thresholds or AcceptanceThresholds()
    failures: list[str] = []

    if metrics.global_similarity < limits.min_global_similarity:
        failures.append(
            "global speaker similarity "
            f"{metrics.global_similarity:.3f} < {limits.min_global_similarity:.3f}"
        )
    if metrics.minimum_window_similarity < limits.min_window_similarity:
        failures.append(
            "minimum short-window speaker similarity "
            f"{metrics.minimum_window_similarity:.3f} < {limits.min_window_similarity:.3f}"
        )
    if metrics.window_similarity_spread > limits.max_window_similarity_spread:
        failures.append(
            "speaker similarity spread across the utterance "
            f"{metrics.window_similarity_spread:.3f} > {limits.max_window_similarity_spread:.3f}"
        )
    if metrics.maximum_adjacent_drop > limits.max_adjacent_window_drop:
        failures.append(
            "adjacent-window identity jump "
            f"{metrics.maximum_adjacent_drop:.3f} > {limits.max_adjacent_window_drop:.3f}"
        )

    if metrics.word_error_rate is None:
        failures.append("word error rate was not measured")
    elif metrics.word_error_rate > limits.max_word_error_rate:
        failures.append(
            f"word error rate {metrics.word_error_rate:.3f} > {limits.max_word_error_rate:.3f}"
        )

    if limits.require_pronunciation_review and metrics.pronunciation_review_passed is not True:
        failures.append("Tamil/English pronunciation review did not pass")
    if metrics.no_garbling_review_passed is not True:
        failures.append("no-garbling listening review did not pass")

    return AcceptanceResult(accepted=not failures, failures=tuple(failures))
