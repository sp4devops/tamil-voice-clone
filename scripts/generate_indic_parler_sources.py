#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥])\s+")
_SOFT_BOUNDARY = re.compile(r"(?<=[,;:])\s+")


def crossfade(parts: list[np.ndarray], sample_rate: int, milliseconds: int = 55) -> np.ndarray:
    if not parts:
        return np.zeros(1, dtype=np.float32)
    result = parts[0].astype(np.float32, copy=False)
    fade = max(1, int(sample_rate * milliseconds / 1000))
    for part in parts[1:]:
        part = part.astype(np.float32, copy=False)
        overlap = min(fade, len(result), len(part))
        if overlap:
            out_curve = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
            in_curve = 1.0 - out_curve
            result = np.concatenate([
                result[:-overlap],
                result[-overlap:] * out_curve + part[:overlap] * in_curve,
                part[overlap:],
            ])
        else:
            result = np.concatenate([result, part])
    return result


def split_text(text: str, max_chars: int = 180) -> list[str]:
    """Split narration into bounded sentence-sized inference units."""
    text = " ".join(text.split()).strip()
    if not text:
        return []

    sentences = [item.strip() for item in _SENTENCE_BOUNDARY.split(text) if item.strip()]
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue

        soft_parts = [item.strip() for item in _SOFT_BOUNDARY.split(sentence) if item.strip()]
        current = ""
        for part in soft_parts:
            candidate = f"{current} {part}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = part
            else:
                current = candidate
        if current:
            chunks.append(current)

    return chunks or [text]


def token_ceiling(text: str, language: str, frame_rate: float) -> int:
    """Allow normal narration duration but prevent an unbounded KV cache."""
    chars_per_second = 8.0 if language == "ta" else 12.0
    estimated_seconds = len(text) / chars_per_second + 1.5
    budget_seconds = estimated_seconds * 1.35
    return max(384, min(1600, math.ceil(frame_rate * budget_seconds)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="ai4bharat/indic-parler-tts")
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    description = payload["description"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
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

    description_inputs = description_tokenizer(description, return_tensors="pt")
    report: dict[str, object] = {
        "model_id": args.model_id,
        "sample_rate": sample_rate,
        "frame_rate": round(frame_rate, 4),
        "seed": args.seed,
        "description": description,
        "attention": "eager",
        "cases": [],
    }

    unit_counter = 0
    for case_index, case in enumerate(payload["cases"]):
        segments: list[np.ndarray] = []
        segment_report: list[dict[str, object]] = []
        for segment_index, segment in enumerate(case["segments"]):
            segment_parts: list[np.ndarray] = []
            unit_reports: list[dict[str, object]] = []
            for unit_index, unit_text in enumerate(split_text(segment["text"])):
                seed = args.seed + case_index * 1000 + segment_index * 100 + unit_index
                torch.manual_seed(seed)
                prompt_inputs = prompt_tokenizer(unit_text, return_tensors="pt")
                max_new_tokens = token_ceiling(unit_text, segment["language"], frame_rate)
                unit_started = time.time()
                with torch.inference_mode():
                    generation = model.generate(
                        input_ids=description_inputs.input_ids,
                        attention_mask=description_inputs.attention_mask,
                        prompt_input_ids=prompt_inputs.input_ids,
                        prompt_attention_mask=prompt_inputs.attention_mask,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                    )
                audio = generation.detach().cpu().float().numpy().squeeze()
                if audio.ndim != 1 or audio.size < sample_rate // 4:
                    raise RuntimeError(
                        f"Invalid generated audio for {case['id']} segment {segment_index} unit {unit_index}"
                    )
                if not np.isfinite(audio).all():
                    raise RuntimeError(
                        f"Non-finite generated audio for {case['id']} segment {segment_index} unit {unit_index}"
                    )

                segment_parts.append(audio.astype(np.float32, copy=False))
                unit_reports.append({
                    "text": unit_text,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "duration_seconds": round(len(audio) / sample_rate, 3),
                    "generation_seconds": round(time.time() - unit_started, 3),
                })
                unit_counter += 1

                del generation, audio, prompt_inputs
                gc.collect()

            segment_audio = crossfade(segment_parts, sample_rate, milliseconds=45)
            segments.append(segment_audio)
            segment_report.append({
                "language": segment["language"],
                "text": segment["text"],
                "duration_seconds": round(len(segment_audio) / sample_rate, 3),
                "units": unit_reports,
            })

        combined = crossfade(segments, sample_rate)
        peak = float(np.max(np.abs(combined)))
        if peak > 0.98:
            combined = combined * (0.98 / peak)
            peak = 0.98
        output_path = args.output_dir / f"{case['id']}.wav"
        sf.write(output_path, combined, sample_rate, subtype="PCM_16")
        report["cases"].append({
            "id": case["id"],
            "output": output_path.name,
            "duration_seconds": round(len(combined) / sample_rate, 3),
            "peak": round(peak, 6),
            "segments": segment_report,
        })

    report["generation_units"] = unit_counter
    report["elapsed_seconds"] = round(time.time() - started, 3)
    (args.output_dir / "parler_source_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
