#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


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

    description_inputs = description_tokenizer(description, return_tensors="pt")
    report: dict[str, object] = {
        "model_id": args.model_id,
        "sample_rate": sample_rate,
        "seed": args.seed,
        "description": description,
        "cases": [],
    }

    for case_index, case in enumerate(payload["cases"]):
        segments: list[np.ndarray] = []
        segment_report: list[dict[str, object]] = []
        for segment_index, segment in enumerate(case["segments"]):
            torch.manual_seed(args.seed + case_index * 100 + segment_index)
            prompt_inputs = prompt_tokenizer(segment["text"], return_tensors="pt")
            segment_started = time.time()
            with torch.inference_mode():
                generation = model.generate(
                    input_ids=description_inputs.input_ids,
                    attention_mask=description_inputs.attention_mask,
                    prompt_input_ids=prompt_inputs.input_ids,
                    prompt_attention_mask=prompt_inputs.attention_mask,
                    do_sample=False,
                )
            audio = generation.detach().cpu().float().numpy().squeeze()
            if audio.ndim != 1 or audio.size < sample_rate // 4:
                raise RuntimeError(f"Invalid generated audio for {case['id']} segment {segment_index}")
            if not np.isfinite(audio).all():
                raise RuntimeError(f"Non-finite generated audio for {case['id']} segment {segment_index}")
            segments.append(audio.astype(np.float32))
            segment_report.append({
                "language": segment["language"],
                "text": segment["text"],
                "duration_seconds": round(len(audio) / sample_rate, 3),
                "generation_seconds": round(time.time() - segment_started, 3),
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

    report["elapsed_seconds"] = round(time.time() - started, 3)
    (args.output_dir / "parler_source_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
