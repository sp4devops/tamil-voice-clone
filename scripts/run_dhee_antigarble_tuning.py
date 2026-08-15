from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel

SAMPLE_RATE = 24000
MODEL_ID = "dheeyantra/dhee-indic-f5"
COMPAT_MODEL_ID = "AbhishekDelMundu/IndicF5"
REF_TEXT = (
    "தோழர் ஸ்டாலின் அவர்கள் தன்னுடைய வேட்புமனு தாக்கல் செய்திருப்பார்ல, "
    "அதுக்குள்ள அவரோட சொத்து விவரங்களையெல்லாம் போட்டிருப்பார்ல."
)

ORIGINAL_TEXT = (
    "தோழர்களே, இன்று production server ரொம்ப slow ஆக இருக்கு. "
    "அதனால் application logs, database connection, மற்றும் network policy "
    "எல்லாத்தையும் கவனமாக check பண்ணுங்க."
)
PHONETIC_TEXT = (
    "தோழர்களே, இன்று புரொடக்ஷன் சர்வர் ரொம்ப ஸ்லோ ஆக இருக்கு. "
    "அதனால் அப்ளிகேஷன் லாக்ஸ், டேட்டாபேஸ் கனெக்ஷன், மற்றும் நெட்வொர்க் பாலிசி "
    "எல்லாத்தையும் கவனமாக செக் பண்ணுங்க."
)
CHUNKS = [
    "தோழர்களே, இன்று production server ரொம்ப slow ஆக இருக்கு.",
    "அதனால் application logs, database connection, மற்றும் network policy எல்லாத்தையும் கவனமாக check பண்ணுங்க.",
]

CONFIG = {
    "architectures": ["INF5Model"],
    "auto_map": {"AutoConfig": "model.INF5Config", "AutoModel": "model.INF5Model"},
    "ckpt_path": "checkpoints/model_best.pt",
    "model_type": "inf5",
    "remove_sil": True,
    "speed": 0.92,
    "cfg_strength": 1.5,
    "nfe_step": 40,
    "sway_sampling_coef": -1.0,
    "target_rms": 0.1,
    "cross_fade_duration": 0.15,
    "torch_dtype": "float32",
    "transformers_version": "4.49.0",
    "vocab_path": "checkpoints/vocab.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anti-garbling tuning from candidate X")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", default="outputs/voice_001_antigarble")
    return parser.parse_args()


def prepare_snapshot(token: str | None) -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            token=token,
            allow_patterns=["model.safetensors", "checkpoints/*", "f5_tts/*", "f5_tts/**/*"],
        )
    )
    model_py = Path(hf_hub_download(repo_id=COMPAT_MODEL_ID, filename="model.py", token=token))
    vocab = Path(
        hf_hub_download(repo_id=COMPAT_MODEL_ID, filename="checkpoints/vocab.txt", token=token)
    )
    local_model = snapshot / "model.py"
    local_vocab = snapshot / "checkpoints" / "vocab.txt"
    local_vocab.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_py, local_model)
    shutil.copyfile(vocab, local_vocab)

    source = local_model.read_text(encoding="utf-8")
    vocab_anchor = (
        'vocab_path = hf_hub_download(config.name_or_path, '
        'filename="checkpoints/vocab.txt")'
    )
    if vocab_anchor not in source:
        raise RuntimeError("Compatible loader vocabulary anchor changed")
    source = source.replace(vocab_anchor, 'vocab_path = os.environ["INDICF5_VOCAB_PATH"]')
    source = source.replace(
        "hf_hub_download(config.name_or_path,",
        "hf_hub_download(os.environ.get('INDICF5_WEIGHTS_REPO', config.name_or_path),",
    )
    inference_anchor = """            speed=self.config.speed,\n            device=self.device,"""
    inference_replacement = """            speed=self.config.speed,\n            nfe_step=getattr(self.config, \"nfe_step\", 40),\n            cfg_strength=getattr(self.config, \"cfg_strength\", 1.5),\n            sway_sampling_coef=getattr(self.config, \"sway_sampling_coef\", -1.0),\n            target_rms=getattr(self.config, \"target_rms\", 0.1),\n            cross_fade_duration=getattr(self.config, \"cross_fade_duration\", 0.15),\n            device=self.device,"""
    if inference_anchor not in source:
        raise RuntimeError("Compatible loader inference anchor changed")
    local_model.write_text(source.replace(inference_anchor, inference_replacement), encoding="utf-8")

    (snapshot / "config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    os.environ["INDICF5_WEIGHTS_REPO"] = MODEL_ID
    os.environ["INDICF5_VOCAB_PATH"] = str(local_vocab.resolve())
    return snapshot


def clean_waveform(audio: object) -> np.ndarray:
    waveform = np.asarray(audio)
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    waveform = waveform.astype(np.float32).reshape(-1)
    if waveform.size < SAMPLE_RATE // 2 or not np.isfinite(waveform).all():
        raise RuntimeError("Invalid waveform")
    if float(np.std(waveform)) < 1e-4 or int(np.unique(np.round(waveform, 6)).size) < 64:
        raise RuntimeError("Silence or DC-only waveform")
    return waveform


def trim_edges(waveform: np.ndarray, threshold: float = 0.006, pad_ms: int = 40) -> np.ndarray:
    frame = max(1, int(0.02 * SAMPLE_RATE))
    energy = np.sqrt(np.convolve(waveform**2, np.ones(frame) / frame, mode="same"))
    active = np.flatnonzero(energy >= threshold)
    if active.size == 0:
        return waveform
    pad = int(pad_ms / 1000 * SAMPLE_RATE)
    start = max(0, int(active[0]) - pad)
    end = min(waveform.size, int(active[-1]) + pad + 1)
    return waveform[start:end]


def generate(model: object, text: str, speed: float, seed: int) -> tuple[np.ndarray, float]:
    model.config.speed = speed
    model.config.cfg_strength = 1.5
    model.config.nfe_step = 40
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    started = time.perf_counter()
    audio = model(text, ref_audio_path=ARGS.reference, ref_text=REF_TEXT)
    return clean_waveform(audio), time.perf_counter() - started


def metrics(waveform: np.ndarray) -> dict[str, float | int]:
    return {
        "duration_seconds": len(waveform) / SAMPLE_RATE,
        "peak": float(np.max(np.abs(waveform))),
        "ac_rms": float(np.std(waveform)),
        "unique_samples": int(np.unique(np.round(waveform, 6)).size),
    }


def main() -> None:
    global ARGS
    ARGS = parse_args()
    output_dir = Path(ARGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None

    load_started = time.perf_counter()
    snapshot = prepare_snapshot(token)
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    load_seconds = time.perf_counter() - load_started

    results: list[dict[str, object]] = []

    # Candidate A: preserve X settings, add clearer punctuation and slower delivery.
    wav_a, elapsed_a = generate(model, ORIGINAL_TEXT, speed=0.90, seed=20260816)
    path_a = output_dir / "a_x_slow_punctuated.wav"
    sf.write(path_a, wav_a, SAMPLE_RATE)
    results.append({"id": path_a.stem, "method": "single_pass_slow", "text": ORIGINAL_TEXT, "generation_seconds": elapsed_a, **metrics(wav_a)})

    # Candidate B: shorter phrase generations, then a clean natural pause.
    chunk_wavs: list[np.ndarray] = []
    chunk_times: list[float] = []
    for index, chunk in enumerate(CHUNKS):
        wav, elapsed = generate(model, chunk, speed=0.92, seed=20260816 + index)
        chunk_wavs.append(trim_edges(wav))
        chunk_times.append(elapsed)
    pause = np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32)
    wav_b = np.concatenate([chunk_wavs[0], pause, chunk_wavs[1]])
    path_b = output_dir / "b_x_chunked_original.wav"
    sf.write(path_b, wav_b, SAMPLE_RATE)
    results.append({"id": path_b.stem, "method": "two_phrase_original_script", "chunks": CHUNKS, "generation_seconds": sum(chunk_times), **metrics(wav_b)})

    # Candidate C: keep one voice path by spelling technical English in Tamil phonetics.
    wav_c, elapsed_c = generate(model, PHONETIC_TEXT, speed=0.92, seed=20260816)
    path_c = output_dir / "c_x_tamil_phonetic.wav"
    sf.write(path_c, wav_c, SAMPLE_RATE)
    results.append({"id": path_c.stem, "method": "single_pass_tamil_phonetic", "text": PHONETIC_TEXT, "generation_seconds": elapsed_c, **metrics(wav_c)})

    report = {
        "voice_id": "voice_001",
        "based_on": "blind candidate X: cfg_strength=1.5, nfe_step=40, exact reference transcript",
        "model_id": MODEL_ID,
        "model_load_seconds": load_seconds,
        "results": results,
        "status": "PENDING_BLIND_HUMAN_GARBLE_REVIEW",
    }
    (output_dir / "antigarble_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ARGS: argparse.Namespace
    main()
