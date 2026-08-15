import numpy as np
import pytest

from tamil_voice_clone.evaluation import (
    AcceptanceThresholds,
    CloneMetrics,
    cosine_similarity,
    evaluate_clone,
)


def test_cosine_similarity_identical_vectors() -> None:
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-6)


def test_accepts_metrics_that_clear_every_gate() -> None:
    metrics = CloneMetrics(
        global_similarity=0.80,
        window_similarities=(0.77, 0.74, 0.76, 0.73),
        word_error_rate=0.03,
        pronunciation_review_passed=True,
        no_garbling_review_passed=True,
    )
    assert evaluate_clone(metrics).accepted is True


def test_rejects_identity_jump_at_language_switch() -> None:
    metrics = CloneMetrics(
        global_similarity=0.78,
        window_similarities=(0.75, 0.73, 0.55, 0.72),
        word_error_rate=0.02,
        pronunciation_review_passed=True,
        no_garbling_review_passed=True,
    )
    result = evaluate_clone(metrics)
    assert result.accepted is False
    assert any("identity jump" in failure for failure in result.failures)


def test_rejects_unmeasured_or_unreviewed_output() -> None:
    metrics = CloneMetrics(global_similarity=0.9, window_similarities=(0.9,))
    result = evaluate_clone(metrics, AcceptanceThresholds())
    assert result.accepted is False
    assert "word error rate was not measured" in result.failures
    assert "Tamil/English pronunciation review did not pass" in result.failures
    assert "no-garbling listening review did not pass" in result.failures
