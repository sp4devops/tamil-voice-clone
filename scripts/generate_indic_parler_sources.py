#!/usr/bin/env python3
"""Generate pronunciation-first Tamil/English source audio with Indic Parler-TTS.

The model is intentionally loaded only in this process. The caller can enforce a
hard RSS limit and delete the model cache before starting the voice-conversion
stage.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥])\s+")
_SOFT_BOUNDARY = re.compile(r"(?<=[,;:])\s+")
_VALID_LANGUAGES = {"ta", "en"}


def _word_wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = word
    chunks.append(current)
    return chunks


def split_text(text: str, max_chars: int = 60) -> list[str]:
    """Split text into bounded, natural inference units without losing words."""
    if max_chars < 20:
        raise ValueError("max_chars must be at least 20")
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []

    units: list[str] = []
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(normalized) if part.strip()]
    for sentence in sentences:
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue
        clauses = [part.strip() for part in _SOFT_BOUNDARY.split(sentence) if part.strip()]
        for clause in clauses:
            if len(clause) <= max_chars:
                units.append(clause)
            else:
                units.extend(_word_wrap(clause, max_chars))

    if not units:
        units = _word_wrap(normalized, max_chars)
    return units


def audio_token_budget(
    text: str,
    language: str,
    frame_rate: float,
    *,
    minimum: int = 160,
    maximum: int = 640,
) -> int:
    """Estimate a bounded audio-token budget for one short speech unit."""
    if language not in _VALID_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid token limits")

    chars_per_second = 7.0 if language == "ta" else 11.0
    estimated_seconds = max(1.5, len(text) / chars_per_second + 0.8)
    budget = math.ceil(estimated_seconds * frame_rate * 1.20)
    return max(minimum, min(maximum, budget))


def join_audio(parts: list[np.ndarray], sample_rate: int, gap_ms: int) -> np.ndarray:
    """Join mono units with a small silence gap to preserve word boundaries."""
    if not parts:
        return np.zeros(1, dtype=np.float32)
    gap = np.zeros(max(0, int(sample_rate * gap_ms / 1000)), dtype=np.float32)
    output: list[np.ndarray] = []
    for index, part in enumerate(parts):
        audio = np.asarray(part, dtype=np.float32).reshape(-1)
        if index and gap.size:
            output.append(gap)
        output.append(audio)
    return np.concatenate(output)


def _malloc_trim() -> None:
    gc.collect()
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _rss_mib() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1024**2, 2)
    except (ImportError, OSError):
        return None


def validate_cases(payload: dict[str, Any]) -> None:
    description = payload.get("description")
    cases = payload.get("cases")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("cases file requires a non-empty description")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases file requires at least one case")

    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        segments = case.get("segments")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"invalid or duplicate case id: {case_id!r}")
        seen.add(case_id)
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"{case_id}: segments must be a non-empty list")
        for segment in segments:
            language = segment.get("language")
            text = segment.get("text")
            if language not in _VALID_LANGUAGES:
                raise ValueError(f"{case_id}: unsupported language {language!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{case_id}: segment text cannot be empty")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="ai4bharat/indic-parler-tts")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-unit-chars", type=int, default=60)
    parser.add_argument("--min-unit-tokens", type=int, default=160)
    parser.add_argument("--max-unit-tokens", type=int, default=640)
    parser.add_argument("--max-unit-seconds", type=float, default=300.0)
    parser.add_argument(
        "--autocast-bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CPU BF16 autocast for activations while retaining FP32 weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    validate_cases(payload)

    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"Loading {args.model_id} in FP32 with low-memory loading", flush=True)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        args.model_id,
        token=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cpu")
    model.eval()

    prompt_tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=True)
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path,
        token=True,
    )
    sample_rate = int(model.config.sampling_rate)
    hop_length = int(model.audio_encoder.config.hop_length)
    frame_rate = sample_rate / hop_length
    description_inputs = description_tokenizer(payload["description"], return_tensors="pt")

    use_bf16 = bool(args.autocast_bf16)
    if use_bf16:
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                _ = torch.ones((2, 2)) @ torch.ones((2, 2))
        except RuntimeError:
            use_bf16 = False
            print("CPU BF16 autocast is unavailable; using FP32 activations", flush=True)

    report: dict[str, Any] = {
        "model_id": args.model_id,
        "sample_rate": sample_rate,
        "frame_rate": round(frame_rate, 4),
        "seed": args.seed,
        "description": payload["description"],
        "model_weights": "float32",
        "activation_autocast": "bfloat16" if use_bf16 else "float32",
        "attention": "eager",
        "limits": {
            "max_unit_chars": args.max_unit_chars,
            "min_unit_tokens": args.min_unit_tokens,
            "max_unit_tokens": args.max_unit_tokens,
            "max_unit_seconds": args.max_unit_seconds,
        },
        "cases": [],
    }

    generation_units = 0
    for case_index, case in enumerate(payload["cases"]):
        print(f"Generating case {case['id']}", flush=True)
        case_segments: list[np.ndarray] = []
        segment_rows: list[dict[str, Any]] = []
        for segment_index, segment in enumerate(case["segments"]):
            unit_texts = split_text(segment["text"], args.max_unit_chars)
            if not unit_texts:
                raise RuntimeError(f"{case['id']}: no inference units produced")
            unit_audio: list[np.ndarray] = []
            unit_rows: list[dict[str, Any]] = []

            for unit_index, unit_text in enumerate(unit_texts):
                seed = args.seed + case_index * 1000 + segment_index * 100 + unit_index
                torch.manual_seed(seed)
                prompt_inputs = prompt_tokenizer(unit_text, return_tensors="pt")
                max_new_tokens = audio_token_budget(
                    unit_text,
                    segment["language"],
                    frame_rate,
                    minimum=args.min_unit_tokens,
                    maximum=args.max_unit_tokens,
                )
                unit_started = time.monotonic()
                print(
                    f"  unit={generation_units + 1} lang={segment['language']} "
                    f"chars={len(unit_text)} max_tokens={max_new_tokens} rss_mib={_rss_mib()} "
                    f"text={unit_text!r}",
                    flush=True,
                )

                autocast_context = (
                    torch.autocast(device_type="cpu", dtype=torch.bfloat16)
                    if use_bf16
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_context:
                    generation = model.generate(
                        input_ids=description_inputs.input_ids,
                        attention_mask=description_inputs.attention_mask,
                        prompt_input_ids=prompt_inputs.input_ids,
                        prompt_attention_mask=prompt_inputs.attention_mask,
                        do_sample=True,
                        temperature=1.0,
                        top_k=50,
                        top_p=0.95,
                        max_new_tokens=max_new_tokens,
                        max_time=args.max_unit_seconds,
                        use_cache=True,
                    )

                audio = generation.detach().cpu().float().numpy().squeeze()
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if audio.size < sample_rate // 4:
                    raise RuntimeError(
                        f"{case['id']} segment {segment_index} unit {unit_index}: output too short"
                    )
                if not np.isfinite(audio).all():
                    raise RuntimeError(
                        f"{case['id']} segment {segment_index} unit {unit_index}: non-finite audio"
                    )

                duration = audio.size / sample_rate
                token_duration_limit = max_new_tokens / frame_rate
                if duration >= token_duration_limit * 0.98:
                    raise RuntimeError(
                        f"{case['id']} segment {segment_index} unit {unit_index}: "
                        "generation reached its token ceiling; refusing likely "
                        "truncated or garbled audio"
                    )

                elapsed = time.monotonic() - unit_started
                unit_audio.append(audio)
                unit_rows.append(
                    {
                        "text": unit_text,
                        "seed": seed,
                        "max_new_tokens": max_new_tokens,
                        "duration_seconds": round(duration, 3),
                        "generation_seconds": round(elapsed, 3),
                        "rss_mib_after": _rss_mib(),
                    }
                )
                generation_units += 1
                del generation, audio, prompt_inputs
                _malloc_trim()

            segment_audio = join_audio(unit_audio, sample_rate, gap_ms=55)
            case_segments.append(segment_audio)
            segment_rows.append(
                {
                    "language": segment["language"],
                    "text": segment["text"],
                    "duration_seconds": round(segment_audio.size / sample_rate, 3),
                    "units": unit_rows,
                }
            )

        combined = join_audio(case_segments, sample_rate, gap_ms=25)
        peak = float(np.max(np.abs(combined)))
        if peak > 0.98:
            combined = combined * (0.98 / peak)
            peak = 0.98
        destination = args.output_dir / f"{case['id']}.wav"
        sf.write(destination, combined, sample_rate, subtype="PCM_24")
        report["cases"].append(
            {
                "id": case["id"],
                "output": destination.name,
                "duration_seconds": round(combined.size / sample_rate, 3),
                "peak": round(peak, 6),
                "segments": segment_rows,
            }
        )
        (args.output_dir / "parler_source_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    report["generation_units"] = generation_units
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["final_rss_mib"] = _rss_mib()
    (args.output_dir / "parler_source_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
