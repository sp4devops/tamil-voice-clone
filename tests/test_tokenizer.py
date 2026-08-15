from pathlib import Path

from tamil_voice_clone.phonemes import PhonemeSpan
from tamil_voice_clone.tokenizer import BOS, EOS, LANGUAGE_IDS, PhonemeVocabulary


def test_vocabulary_encodes_language_ids() -> None:
    spans = [
        PhonemeSpan(language="ta", source_text="மச்சான்", ipa="matʃaːn"),
        PhonemeSpan(language="en", source_text="server", ipa="sɜːvə"),
    ]
    vocabulary = PhonemeVocabulary.from_spans([spans])
    encoded = vocabulary.encode(spans)

    assert encoded.token_ids[0] == vocabulary.to_id[BOS]
    assert encoded.token_ids[-1] == vocabulary.to_id[EOS]
    assert encoded.language_ids[0] == 0
    assert encoded.language_ids[-1] == 0
    assert LANGUAGE_IDS["ta"] in encoded.language_ids
    assert LANGUAGE_IDS["en"] in encoded.language_ids
    assert len(encoded.token_ids) == len(encoded.language_ids)


def test_vocabulary_round_trip(tmp_path: Path) -> None:
    spans = [PhonemeSpan(language="ta", source_text="ரொம்ப", ipa="romba")]
    vocabulary = PhonemeVocabulary.from_spans([spans])
    path = tmp_path / "phonemes.json"
    vocabulary.save(path)

    loaded = PhonemeVocabulary.load(path)
    assert loaded.symbols == vocabulary.symbols
    assert loaded.to_id == vocabulary.to_id
