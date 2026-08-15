from tamil_voice_clone.text import normalize_text


def test_detects_native_script_code_mix() -> None:
    profile = normalize_text("இன்னிக்கு Kubernetes cluster slow")
    assert profile.has_tamil
    assert profile.has_latin
    assert profile.is_code_mixed


def test_normalizes_common_devops_terms() -> None:
    profile = normalize_text("rabbitmq and mongodb issue")
    assert profile.normalized == "Rabbit M Q and Mongo D B issue"


def test_rejects_empty_text() -> None:
    try:
        normalize_text("   ")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")
