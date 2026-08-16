from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parler_generator_is_not_truncated() -> None:
    path = ROOT / "scripts/generate_indic_parler_sources.py"
    source = path.read_text(encoding="utf-8")
    assert len(source) > 8_000
    compile(source, str(path), "exec")


def test_split_text_preserves_all_words_and_bounds_units() -> None:
    generator = load_script("generate_indic_parler_sources")
    text = (
        "ஒரு பெரிய கணினி அமைப்பில் பிரச்சினை ஏற்பட்டால் உடனடியாக முடிவுக்கு வரக்கூடாது. "
        "முதலில் பதிவுகளை கவனமாக பார்க்க வேண்டும்."
    )
    units = generator.split_text(text, max_chars=45)
    assert units
    assert all(len(unit) <= 45 or len(unit.split()) == 1 for unit in units)

    normalized_original = " ".join(text.split()).replace(". ", ". ")
    reconstructed = " ".join(units)
    for word in normalized_original.split():
        assert word in reconstructed


def test_audio_token_budget_is_bounded() -> None:
    generator = load_script("generate_indic_parler_sources")
    assert generator.audio_token_budget("hello", "en", 86.13) >= 160
    assert generator.audio_token_budget("அ" * 5_000, "ta", 86.13) == 640
    with pytest.raises(ValueError):
        generator.audio_token_budget("hello", "xx", 86.13)


def test_cases_schema_is_valid() -> None:
    generator = load_script("generate_indic_parler_sources")
    path = ROOT / "eval/voice_001_parler_seed_cases.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    generator.validate_cases(payload)
    assert len(payload["cases"]) == 4


def test_seed_vc_wrapper_forces_float32_on_cpu() -> None:
    source = (ROOT / "scripts/run_seed_vc_batch.py").read_text(encoding="utf-8")
    assert 'dtype = torch.float16 if device.type == "cuda" else torch.float32' in source
    assert "reference_info.duration < 20.0" in source
    assert "from inference_v2 import" not in source


def test_memory_watchdog_has_timeout_and_process_group_cleanup() -> None:
    source = (ROOT / "scripts/run_with_memory_cap.py").read_text(encoding="utf-8")
    assert "--timeout-seconds" in source
    assert "start_new_session=True" in source
    assert "os.killpg" in source
    compile(source, "run_with_memory_cap.py", "exec")
