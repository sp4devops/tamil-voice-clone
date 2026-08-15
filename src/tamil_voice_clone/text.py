import re
import unicodedata
from dataclasses import dataclass

_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_SPACE_RE = re.compile(r"\s+")

_TECH_NORMALIZATIONS = {
    "rabbitmq": "Rabbit M Q",
    "mongodb": "Mongo D B",
    "postgresql": "Postgre S Q L",
    "kubectl": "kube control",
    "k8s": "Kubernetes",
}


@dataclass(frozen=True)
class TextProfile:
    normalized: str
    has_tamil: bool
    has_latin: bool

    @property
    def is_code_mixed(self) -> bool:
        return self.has_tamil and self.has_latin


def normalize_text(text: str) -> TextProfile:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    normalized = unicodedata.normalize("NFC", text.strip())
    normalized = _SPACE_RE.sub(" ", normalized)

    for source, replacement in _TECH_NORMALIZATIONS.items():
        normalized = re.sub(
            rf"\b{re.escape(source)}\b",
            replacement,
            normalized,
            flags=re.IGNORECASE,
        )

    return TextProfile(
        normalized=normalized,
        has_tamil=bool(_TAMIL_RE.search(normalized)),
        has_latin=bool(_LATIN_RE.search(normalized)),
    )
