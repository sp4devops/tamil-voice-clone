from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime limits for low-memory CPU inference."""

    sample_rate: int = 22050
    min_reference_seconds: float = 20.0
    preferred_reference_seconds: float = 30.0
    max_reference_seconds: float = 120.0
    max_rss_mb: int = 2048
    target_rss_mb: int = 1536
    intra_op_threads: int = 2
    inter_op_threads: int = 1
