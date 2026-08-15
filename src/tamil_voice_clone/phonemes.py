from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .text import normalize_text

LanguageCode = Literal["ta", "en"]

_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
_TOKEN_RE = re.compile(r"[\u0B80-\u0BFFA-Za-z0-9_+.-]+|[^\w\s]", re.UNICODE)

# Small built-in seed lexicon. This is intentionally conservative: unknown Latin
# words remain English until a better transliterator or user lexicon identifies
# them. Wrongly forcing an English technical word through Tamil G2P is worse.
_TANGLISH_TO_TAMIL = {
    "aagiduchu": "ஆகிடுச்சு",
    "aagiduthu": "ஆகிடுது",
    "enna": "என்ன",
    "epdi": "எப்படி",
    "inniku": "இன்னிக்கு",
    "iruku": "இருக்கு",
    "irukku": "இருக்கு",
    "konjam": "கொஞ்சம்",
    "machan": "மச்சான்",
    "makkale": "மக்களே",
    "naalaiku": "நாளைக்கு",
    "panniten": "பண்ணிட்டேன்",
    "pannunga": "பண்ணுங்க",
    "puriyala": "புரியல",
    "romba": "ரொம்ப",
}


class PhonemizerError(RuntimeError):
    pass


@dataclass(frozen=True)
class LanguageSpan:
    language: LanguageCode
    text: str


@dataclass(frozen=True)
class PhonemeSpan:
    language: LanguageCode
    source_text: str
    ipa: str


def _token_language(token: str) -> tuple[LanguageCode, str]:
    if _TAMIL_RE.search(token):
        return "ta", token

    mapped = _TANGLISH_TO_TAMIL.get(token.casefold())
    if mapped is not None:
        return "ta", mapped

    return "en", token


def split_language_spans(text: str) -> list[LanguageSpan]:
    """Split Tamil/English/Tanglish text into contiguous G2P spans.

    Latin-script Tanglish is only converted when it is in the conservative
    built-in lexicon. This prevents technical words such as Kubernetes and
    MongoDB from accidentally going through Tamil G2P.
    """
    profile = normalize_text(text)
    tokens = _TOKEN_RE.findall(profile.normalized)
    if not tokens:
        raise ValueError("Text does not contain phonemizable tokens.")

    spans: list[LanguageSpan] = []
    current_language: LanguageCode | None = None
    current_tokens: list[str] = []

    def flush() -> None:
        nonlocal current_language, current_tokens
        if current_language is not None and current_tokens:
            spans.append(LanguageSpan(current_language, " ".join(current_tokens)))
        current_language = None
        current_tokens = []

    for token in tokens:
        language, normalized_token = _token_language(token)
        if current_language is None:
            current_language = language
        elif language != current_language:
            flush()
            current_language = language
        current_tokens.append(normalized_token)

    flush()
    return spans


class EspeakPhonemizer:
    """Small system-eSpeak frontend that emits language-tagged IPA spans."""

    def __init__(
        self,
        executable: str | Path = "espeak-ng",
        tamil_voice: str = "ta",
        english_voice: str = "en",
        timeout_seconds: float = 10.0,
    ) -> None:
        requested = str(executable)
        resolved = shutil.which(requested) if Path(requested).name == requested else requested
        if not resolved or not Path(resolved).exists():
            raise PhonemizerError(
                "espeak-ng was not found. Install eSpeak-NG and ensure `espeak-ng` is on PATH."
            )
        self.executable = resolved
        self.voices: dict[LanguageCode, str] = {"ta": tamil_voice, "en": english_voice}
        self.timeout_seconds = timeout_seconds

    def phonemize_span(self, span: LanguageSpan) -> PhonemeSpan:
        command = [
            self.executable,
            "-q",
            "--ipa",
            "--sep=|",
            "-v",
            self.voices[span.language],
            span.text,
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise PhonemizerError(
                f"eSpeak-NG failed for {span.language!r} span: {span.text!r}"
            ) from exc

        ipa = " ".join(result.stdout.split())
        if not ipa:
            raise PhonemizerError(
                f"eSpeak-NG returned no phonemes for {span.language!r} span: {span.text!r}"
            )
        return PhonemeSpan(language=span.language, source_text=span.text, ipa=ipa)

    def phonemize(self, text: str) -> list[PhonemeSpan]:
        return [self.phonemize_span(span) for span in split_language_spans(text)]


def tagged_phoneme_text(spans: list[PhonemeSpan]) -> str:
    """Serialize phonemes for training while preserving language boundaries."""
    return " ".join(f"<{span.language}> {span.ipa}" for span in spans)
