from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .phonemes import PhonemeSpan

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = (PAD, BOS, EOS, UNK)
LANGUAGE_IDS = {"ta": 1, "en": 2}


@dataclass(frozen=True)
class TokenizedPhonemes:
    token_ids: list[int]
    language_ids: list[int]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.language_ids):
            raise ValueError("token_ids and language_ids must have the same length")


class PhonemeVocabulary:
    """Small deterministic character vocabulary for IPA produced by the frontend.

    We use Unicode IPA characters rather than word pieces so Tamil and English
    share acoustic symbols naturally. Spaces are retained because they carry
    useful word-boundary information for duration/prosody learning.
    """

    def __init__(self, symbols: list[str]) -> None:
        ordered: list[str] = []
        seen: set[str] = set()
        for symbol in [*SPECIAL_TOKENS, *symbols]:
            normalized = unicodedata.normalize("NFC", symbol)
            if normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
        self.symbols = ordered
        self.to_id = {symbol: index for index, symbol in enumerate(self.symbols)}

    @classmethod
    def from_spans(cls, utterances: list[list[PhonemeSpan]]) -> "PhonemeVocabulary":
        symbols: set[str] = set()
        for spans in utterances:
            for span in spans:
                ipa = unicodedata.normalize("NFC", span.ipa)
                symbols.update(ipa)
        return cls(sorted(symbols))

    def encode(self, spans: list[PhonemeSpan]) -> TokenizedPhonemes:
        token_ids = [self.to_id[BOS]]
        language_ids = [0]

        for span in spans:
            language_id = LANGUAGE_IDS[span.language]
            for symbol in unicodedata.normalize("NFC", span.ipa):
                token_ids.append(self.to_id.get(symbol, self.to_id[UNK]))
                language_ids.append(language_id)

        token_ids.append(self.to_id[EOS])
        language_ids.append(0)
        return TokenizedPhonemes(token_ids=token_ids, language_ids=language_ids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": 1, "symbols": self.symbols}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "PhonemeVocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != 1:
            raise ValueError("Unsupported phoneme vocabulary version")
        return cls([str(symbol) for symbol in payload["symbols"]])
