from tamil_voice_clone.phonemes import LanguageSpan, split_language_spans


def test_split_tamil_and_english_spans() -> None:
    spans = split_language_spans("மச்சான் Kubernetes slow ஆகிடுச்சு")
    assert spans == [
        LanguageSpan("ta", "மச்சான்"),
        LanguageSpan("en", "Kubernetes slow"),
        LanguageSpan("ta", "ஆகிடுச்சு"),
    ]


def test_known_tanglish_is_mapped_to_tamil() -> None:
    spans = split_language_spans("Machan server down aagiduchu")
    assert spans == [
        LanguageSpan("ta", "மச்சான்"),
        LanguageSpan("en", "server down"),
        LanguageSpan("ta", "ஆகிடுச்சு"),
    ]


def test_unknown_latin_word_stays_english() -> None:
    spans = split_language_spans("Prometheus romba slow")
    assert spans == [
        LanguageSpan("en", "Prometheus"),
        LanguageSpan("ta", "ரொம்ப"),
        LanguageSpan("en", "slow"),
    ]
