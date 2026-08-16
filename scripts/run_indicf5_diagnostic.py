from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import time
import unicodedata
from contextlib import redirect_stderr, redirect_stdout
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from safetensors.torch import load_file

from run_dhee_baseline import prepare_compatible_snapshot, validate_waveform

SAMPLE_RATE = 24000
MODEL_ID = "dheeyantra/dhee-indic-f5"
COMPAT_MODEL_ID = "AbhishekDelMundu/IndicF5"
DIAGNOSTIC_TEXT = "இது தெளிவான தமிழ் குரல்."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secure low-cost IndicF5 diagnostic")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-text-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def safe_normalize(text: str) -> str:
    return "".join(
        ch.casefold()
        for ch in text
        if unicodedata.category(ch)[0] in {"L", "N", "M"}
    )


def tamil_clean(text: str) -> str:
    return "".join(re.findall(r"[\u0B80-\u0BFF]", text))


def tamil_letter_ratio(text: str) -> float:
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    if not letters:
        return 0.0
    tamil_letters = [ch for ch in letters if "\u0B80" <= ch <= "\u0BFF"]
    return len(tamil_letters) / len(letters)


def transcribe_private(verifier: WhisperModel, path: Path, beam_size: int) -> str:
    segments, _ = verifier.transcribe(
        str(path),
        language="ta",
        beam_size=beam_size,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def resolve_reference_text(
    verifier: WhisperModel,
    reference: Path,
    protected_text: str,
) -> tuple[dict[str, object], str]:
    """Choose a transcript without ever exposing transcript contents.

    The encrypted transcript remains preferred when it agrees with the audio.
    If it does not, derive a private transcript directly from the same reference
    clip and require two deterministic Whisper decodes to agree before using it.
    The derived text exists only in memory for this workflow run.
    """

    info = sf.info(reference)
    primary = transcribe_private(verifier, reference, beam_size=5)
    secondary = transcribe_private(verifier, reference, beam_size=1)

    protected_norm = safe_normalize(protected_text)
    primary_norm = safe_normalize(primary)
    secondary_norm = safe_normalize(secondary)

    protected_similarity = (
        SequenceMatcher(None, protected_norm, primary_norm).ratio()
        if protected_norm and primary_norm
        else 0.0
    )
    protected_length_ratio = len(primary_norm) / max(1, len(protected_norm))
    protected_accepted = bool(
        3.0 <= float(info.duration) <= 10.0
        and len(protected_norm) >= 8
        and len(primary_norm) >= 8
        and protected_similarity >= 0.82
        and 0.70 <= protected_length_ratio <= 1.35
    )

    consensus_similarity = (
        SequenceMatcher(None, primary_norm, secondary_norm).ratio()
        if primary_norm and secondary_norm
        else 0.0
    )
    derived_script_ratio = tamil_letter_ratio(primary)
    derived_token_count = len(primary.split())
    derived_accepted = bool(
        3.0 <= float(info.duration) <= 10.0
        and 10 <= len(primary_norm) <= 160
        and 2 <= derived_token_count <= 24
        and consensus_similarity >= 0.88
        and derived_script_ratio >= 0.75
    )

    if protected_accepted:
        selected_text = protected_text
        mode = "PROTECTED_EXACT_TRANSCRIPT"
        accepted = True
    elif derived_accepted:
        selected_text = primary
        mode = "PRIVATE_ASR_DERIVED_TRANSCRIPT"
        accepted = True
    else:
        selected_text = ""
        mode = "REJECTED"
        accepted = False

    metrics: dict[str, object] = {
        "duration_seconds": round(float(info.duration), 3),
        "protected_char_count": len(protected_norm),
        "protected_token_count": len(protected_text.split()),
        "private_asr_char_count": len(primary_norm),
        "private_asr_token_count": derived_token_count,
        "protected_alignment_similarity": round(protected_similarity, 4),
        "protected_alignment_length_ratio": round(protected_length_ratio, 4),
        "protected_alignment_accepted": protected_accepted,
        "private_asr_consensus_similarity": round(consensus_similarity, 4),
        "private_asr_tamil_script_ratio": round(derived_script_ratio, 4),
        "private_asr_accepted": derived_accepted,
        "accepted": accepted,
        "mode": mode,
    }
    print("reference_alignment=" + json.dumps(metrics, sort_keys=True))
    return metrics, selected_text


def load_compat_model_direct(snapshot: Path):
    """Instantiate the compatibility model without a second HF weight reload.

    The compatibility model's __init__ already loads the Dhee safetensors into
    the EMA transformer. Calling AutoModel.from_pretrained afterwards triggers
    another generic Transformers state-dict pass, which is unnecessary and was
    the source of the earlier `_orig_mod`/EMA warning noise.
    """

    model_path = snapshot / "model.py"
    spec = importlib.util.spec_from_file_location("voice001_indicf5_compat", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import IndicF5 compatibility model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config_payload = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config = module.INF5Config(**config_payload)
    config._name_or_path = MODEL_ID

    captured = io.StringIO()
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            model = module.INF5Model(config)
    except Exception as exc:
        raise RuntimeError(
            f"Direct IndicF5 compatibility construction failed: {type(exc).__name__}"
        ) from None
    return model


def unwrap(module):
    return module._orig_mod if hasattr(module, "_orig_mod") else module


def split_checkpoint(snapshot: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    raw = load_file(str(snapshot / "model.safetensors"), device="cpu")
    ema: dict[str, torch.Tensor] = {}
    vocoder: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if key.startswith("ema_model._orig_mod."):
            ema[key.removeprefix("ema_model._orig_mod.")] = value
        elif key.startswith("ema_model."):
            ema[key.removeprefix("ema_model.")] = value
        elif key.startswith("vocoder._orig_mod."):
            vocoder[key.removeprefix("vocoder._orig_mod.")] = value
        elif key.startswith("vocoder."):
            vocoder[key.removeprefix("vocoder.")] = value
    del raw
    if not ema:
        raise RuntimeError("No EMA transformer weights found in Dhee checkpoint")
    return ema, vocoder


def tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def weight_probe(module, expected: dict[str, torch.Tensor]) -> dict[str, object]:
    runtime = module.state_dict()
    common = [
        key
        for key, value in expected.items()
        if key in runtime
        and tuple(runtime[key].shape) == tuple(value.shape)
        and torch.is_floating_point(value)
        and value.numel() >= 4096
    ]
    if not common:
        raise RuntimeError("No representative runtime parameters match checkpoint structure")
    chosen = sorted(common, key=lambda key: (-expected[key].numel(), key))[:3]
    samples = []
    for key in chosen:
        runtime_hash = tensor_digest(runtime[key])
        checkpoint_hash = tensor_digest(expected[key])
        samples.append(
            {
                "parameter": key,
                "runtime_checksum": runtime_hash,
                "checkpoint_checksum": checkpoint_hash,
                "match": runtime_hash == checkpoint_hash,
            }
        )
    return {
        "common_parameter_count": len(common),
        "sample_count": len(samples),
        "all_sampled_match": all(bool(item["match"]) for item in samples),
        "samples": samples,
    }


def repair_and_prove_weights(model, snapshot: Path) -> dict[str, object]:
    ema_expected, vocoder_expected = split_checkpoint(snapshot)
    ema_runtime = unwrap(model.ema_model)
    vocoder_runtime = unwrap(model.vocoder)

    before_ema = weight_probe(ema_runtime, ema_expected)
    before_vocoder = weight_probe(vocoder_runtime, vocoder_expected) if vocoder_expected else None

    ema_missing, ema_unexpected = ema_runtime.load_state_dict(ema_expected, strict=False)
    vocoder_missing: list[str] = []
    vocoder_unexpected: list[str] = []
    if vocoder_expected:
        vocoder_missing, vocoder_unexpected = vocoder_runtime.load_state_dict(
            vocoder_expected, strict=False
        )

    after_ema = weight_probe(ema_runtime, ema_expected)
    after_vocoder = weight_probe(vocoder_runtime, vocoder_expected) if vocoder_expected else None

    metrics: dict[str, object] = {
        "ema_loaded_by_constructor": bool(before_ema["all_sampled_match"]),
        "ema_before": before_ema,
        "ema_after": after_ema,
        "ema_missing_after_repair": len(ema_missing),
        "ema_unexpected_after_repair": len(ema_unexpected),
        "vocoder_checkpoint_present": bool(vocoder_expected),
        "vocoder_before": before_vocoder,
        "vocoder_after": after_vocoder,
        "vocoder_missing_after_repair": len(vocoder_missing),
        "vocoder_unexpected_after_repair": len(vocoder_unexpected),
        "effective_weights_verified": bool(
            after_ema["all_sampled_match"]
            and (after_vocoder is None or after_vocoder["all_sampled_match"])
            and len(ema_unexpected) == 0
            and len(vocoder_unexpected) == 0
        ),
    }
    print("weight_validation=" + json.dumps(metrics, sort_keys=True))
    return metrics


def verify_generated(verifier: WhisperModel, path: Path) -> dict[str, float | int | bool]:
    asr_text = transcribe_private(verifier, path, beam_size=5)
    expected = tamil_clean(DIAGNOSTIC_TEXT)
    observed = tamil_clean(asr_text)
    overall = SequenceMatcher(None, expected, observed).ratio() if expected else 0.0
    prefix_len = max(5, round(len(expected) * 0.50))
    prefix = SequenceMatcher(
        None, expected[:prefix_len], observed[:prefix_len]
    ).ratio() if observed else 0.0
    length_ratio = len(observed) / max(1, len(expected))
    duration = float(sf.info(path).duration)
    script_ratio = tamil_letter_ratio(asr_text)
    accepted = bool(
        0.8 <= duration <= 5.0
        and overall >= 0.90
        and prefix >= 0.90
        and 0.80 <= length_ratio <= 1.25
        and script_ratio >= 0.85
    )
    return {
        "duration_seconds": round(duration, 3),
        "asr_tamil_char_count": len(observed),
        "target_tamil_char_count": len(expected),
        "overall_asr_similarity": round(overall, 4),
        "prefix_asr_similarity": round(prefix, 4),
        "asr_length_ratio": round(length_ratio, 4),
        "tamil_script_letter_ratio": round(script_ratio, 4),
        "accepted": accepted,
    }


def main() -> None:
    args = parse_args()
    reference = Path(args.reference)
    protected_text = Path(args.reference_text_file).read_text(encoding="utf-8").strip()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verifier = WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )

    alignment, effective_ref_text = resolve_reference_text(
        verifier, reference, protected_text
    )
    if not bool(alignment["accepted"]):
        raise SystemExit("Reference transcript alignment/consensus gate failed")

    snapshot = prepare_compatible_snapshot(
        MODEL_ID,
        COMPAT_MODEL_ID,
        os.environ.get("HF_TOKEN") or None,
    )
    started = time.perf_counter()
    model = load_compat_model_direct(snapshot)
    model_load_seconds = time.perf_counter() - started

    weights = repair_and_prove_weights(model, snapshot)
    if not weights["effective_weights_verified"]:
        raise SystemExit("Effective model weight validation failed")

    torch.manual_seed(7)
    np.random.seed(7)
    started = time.perf_counter()
    captured = io.StringIO()
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            generated = model(
                DIAGNOSTIC_TEXT,
                ref_audio_path=str(reference),
                ref_text=effective_ref_text,
            )
    except Exception as exc:
        raise RuntimeError(f"IndicF5 generation failed: {type(exc).__name__}") from None
    generation_seconds = time.perf_counter() - started

    waveform = np.asarray(generated)
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    waveform = waveform.astype(np.float32).reshape(-1)
    signal = validate_waveform(waveform, "diagnostic_direct_short")

    candidate = output_dir / "diagnostic_direct_short.wav"
    sf.write(candidate, waveform, SAMPLE_RATE, subtype="PCM_16")
    generation = verify_generated(verifier, candidate)

    report = {
        "status": "PENDING_HUMAN_LISTENING_REVIEW" if generation["accepted"] else "REJECTED_STRICT_ASR_GATE",
        "strategy": "OFFICIAL_DIRECT_SHORT_PHRASE",
        "api_contract": "generation_text_separate_from_reference_text",
        "reference_text_source": alignment["mode"],
        "model_id": MODEL_ID,
        "reference_alignment": alignment,
        "weight_validation": weights,
        "model_load_seconds": round(model_load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "signal_metrics": signal,
        "generation_metrics": generation,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("safe_diagnostic=" + json.dumps(report, ensure_ascii=True, sort_keys=True))

    if not generation["accepted"]:
        candidate.unlink(missing_ok=True)
        raise SystemExit("Diagnostic synthesis rejected by strict ASR gate")


if __name__ == "__main__":
    main()
