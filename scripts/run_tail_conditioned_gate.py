from __future__ import annotations

import argparse
import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from transformers import AutoModel

from run_dhee_baseline import prepare_compatible_snapshot, validate_waveform

SAMPLE_RATE = 24000
TARGET_TEXT = "இது தெளிவான இயல்பான தமிழ் குரல்."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-text-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def tamil_clean(text: str) -> str:
    return "".join(re.findall(r"[\u0B80-\u0BFF]", text))


def transcribe_words(verifier: WhisperModel, path: Path) -> list[dict[str, float | str]]:
    segments, _ = verifier.transcribe(
        str(path),
        language="ta",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0,
    )
    words: list[dict[str, float | str]] = []
    for segment in segments:
        for word in segment.words or []:
            clean = tamil_clean(word.word)
            if clean:
                words.append(
                    {
                        "clean": clean,
                        "start": float(word.start),
                        "end": float(word.end),
                    }
                )
    return words


def transcribe_clean(verifier: WhisperModel, path: Path) -> str:
    segments, _ = verifier.transcribe(
        str(path),
        language="ta",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0,
    )
    return tamil_clean(" ".join(segment.text.strip() for segment in segments))


def locate_tail(
    words: list[dict[str, float | str]], chunk_text: str, duration: float
) -> dict[str, float] | None:
    expected = tamil_clean(chunk_text)
    expected_words = [
        tamil_clean(piece) for piece in chunk_text.split() if tamil_clean(piece)
    ]
    best: dict[str, float] | None = None
    lower = max(1, len(expected_words) - 1)
    upper = len(expected_words) + 1
    first = max(0, len(words) - 10)

    for start in range(first, len(words)):
        for size in range(lower, upper + 1):
            end = start + size
            if end > len(words):
                continue
            candidate = "".join(str(word["clean"]) for word in words[start:end])
            similarity = SequenceMatcher(None, expected, candidate).ratio()
            tail_position = float(words[end - 1]["end"]) / max(duration, 0.001)
            rank = (
                similarity
                + 0.12 * min(1.0, tail_position)
                - 0.025 * abs(size - len(expected_words))
            )
            item = {
                "start": float(words[start]["start"]),
                "end": float(words[end - 1]["end"]),
                "similarity": similarity,
                "tail_position": tail_position,
                "rank": rank,
            }
            if best is None or rank > best["rank"]:
                best = item
    return best


def generate_chunk(
    model,
    verifier: WhisperModel,
    reference: str,
    reference_text: str,
    chunk: dict[str, str],
    private_dir: Path,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    expected = tamil_clean(chunk["text"])

    for seed in (71, 7):
        started = time.perf_counter()
        full_path = private_dir / f"{chunk['id']}_carrier_{seed}.wav"
        crop_path = private_dir / f"{chunk['id']}_crop_{seed}.wav"

        torch.manual_seed(seed)
        np.random.seed(seed)
        generated = model(
            chunk["carrier"],
            ref_audio_path=reference,
            ref_text=reference_text,
        )
        waveform = np.asarray(generated)
        if waveform.dtype == np.int16:
            waveform = waveform.astype(np.float32) / 32768.0
        waveform = waveform.astype(np.float32).reshape(-1)
        signal = validate_waveform(waveform, f"{chunk['id']}_seed_{seed}")
        duration = len(waveform) / SAMPLE_RATE
        sf.write(full_path, waveform, SAMPLE_RATE, subtype="PCM_16")

        words = transcribe_words(verifier, full_path)
        match = locate_tail(words, chunk["text"], duration)
        result: dict[str, object] = {
            "chunk": chunk["id"],
            "seed": seed,
            "carrier_duration_seconds": round(duration, 3),
            "word_count": len(words),
            "accepted": False,
            "generation_seconds": round(time.perf_counter() - started, 3),
            **signal,
        }

        if match is None:
            result["rejection"] = "NO_TAMIL_WORD_TIMESTAMPS"
            attempts.append(result)
            print(f"chunk={chunk['id']} seed={seed} rejected=no-word-timestamps")
            continue

        crop_start = max(0.0, match["start"] - 0.045)
        crop_end = min(duration, match["end"] + 0.085)
        cropped = waveform[
            int(crop_start * SAMPLE_RATE) : int(crop_end * SAMPLE_RATE)
        ]
        if len(cropped) < int(0.35 * SAMPLE_RATE):
            result["rejection"] = "CROP_TOO_SHORT"
            attempts.append(result)
            print(f"chunk={chunk['id']} seed={seed} rejected=crop-too-short")
            continue

        sf.write(crop_path, cropped, SAMPLE_RATE, subtype="PCM_16")
        crop_clean = transcribe_clean(verifier, crop_path)
        crop_similarity = SequenceMatcher(None, expected, crop_clean).ratio()
        crop_length_ratio = len(crop_clean) / max(1, len(expected))
        accepted = (
            match["similarity"] >= 0.84
            and match["tail_position"] >= 0.72
            and crop_similarity >= 0.88
            and 0.75 <= crop_length_ratio <= 1.30
        )
        result.update(
            {
                "tail_match_similarity": round(match["similarity"], 4),
                "tail_position": round(match["tail_position"], 4),
                "crop_similarity": round(crop_similarity, 4),
                "crop_length_ratio": round(crop_length_ratio, 4),
                "crop_duration_seconds": round(len(cropped) / SAMPLE_RATE, 3),
                "accepted": accepted,
            }
        )
        if not accepted:
            result["rejection"] = "TAIL_CHUNK_GATE"
        attempts.append(result)
        print(
            f"chunk={chunk['id']} seed={seed} "
            f"tail={match['similarity']:.3f} crop={crop_similarity:.3f} "
            f"position={match['tail_position']:.3f} accepted={accepted}"
        )
        if accepted:
            return cropped, result, attempts

    raise RuntimeError(f"No clear tail-conditioned output for {chunk['id']}")


def crossfade(parts: list[np.ndarray]) -> np.ndarray:
    combined = parts[0]
    for part in parts[1:]:
        fade = min(int(0.035 * SAMPLE_RATE), len(combined) // 4, len(part) // 4)
        if fade <= 0:
            combined = np.concatenate([combined, part])
            continue
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        overlap = combined[-fade:] * (1.0 - ramp) + part[:fade] * ramp
        combined = np.concatenate([combined[:-fade], overlap, part[fade:]])
    return combined


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = Path(args.reference).parent
    reference_text = Path(args.reference_text_file).read_text(encoding="utf-8").strip()

    snapshot = prepare_compatible_snapshot(
        "dheeyantra/dhee-indic-f5",
        "AbhishekDelMundu/IndicF5",
        os.environ.get("HF_TOKEN") or None,
    )
    model = AutoModel.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True
    )
    verifier = WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )

    chunks = [
        {
            "id": "part1",
            "text": "இது தெளிவான",
            "carrier": "இது தெளிவான இயல்பான தமிழ் குரல். இது தெளிவான.",
        },
        {
            "id": "part2",
            "text": "இயல்பான தமிழ் குரல்",
            "carrier": "இது தெளிவான இயல்பான தமிழ் குரல். இயல்பான தமிழ் குரல்.",
        },
    ]

    parts: list[np.ndarray] = []
    selected: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    for chunk in chunks:
        audio, metrics, chunk_attempts = generate_chunk(
            model,
            verifier,
            args.reference,
            reference_text,
            chunk,
            private_dir,
        )
        parts.append(audio)
        selected.append(metrics)
        attempts.extend(chunk_attempts)

    combined = crossfade(parts)
    output_path = output_dir / "tamil_exact_reference_gate.wav"
    sf.write(output_path, combined, SAMPLE_RATE, subtype="PCM_16")

    target_clean = tamil_clean(TARGET_TEXT)
    prefix_len = max(8, round(len(target_clean) * 0.55))
    final_clean = transcribe_clean(verifier, output_path)
    overall = SequenceMatcher(None, target_clean, final_clean).ratio()
    prefix = SequenceMatcher(
        None, target_clean[:prefix_len], final_clean[:prefix_len]
    ).ratio()
    length_ratio = len(final_clean) / max(1, len(target_clean))
    duration = len(combined) / SAMPLE_RATE
    accepted = (
        1.5 <= duration <= 7.0
        and 0.82 <= length_ratio <= 1.22
        and overall >= 0.90
        and prefix >= 0.88
    )

    report = {
        "strategy": "TAIL_CONDITIONED_CHUNK_SYNTHESIS",
        "target_text": TARGET_TEXT,
        "chunk_thresholds": {"tail_match": 0.84, "cropped_chunk": 0.88},
        "final_thresholds": {"overall": 0.90, "prefix": 0.88},
        "attempts": attempts,
        "selected_chunks": selected,
        "final_metrics": {
            "duration_seconds": round(duration, 3),
            "overall_asr_similarity": round(overall, 4),
            "prefix_asr_similarity": round(prefix, 4),
            "length_ratio": round(length_ratio, 4),
            "accepted": accepted,
        },
    }

    if not accepted:
        report["status"] = "REJECTED_FINAL_ASSEMBLY_GATE"
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_path.unlink(missing_ok=True)
        raise SystemExit(
            "Tail chunks passed separately, but final assembly failed the strict gate"
        )

    report["status"] = "PENDING_HUMAN_LISTENING_REVIEW"
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"final duration={duration:.2f} overall={overall:.3f} "
        f"prefix={prefix:.3f} length={length_ratio:.3f}"
    )


if __name__ == "__main__":
    main()
