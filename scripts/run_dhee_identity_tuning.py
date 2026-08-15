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
TARGET_TEXT = (
    "தோழர்களே, இன்று production server ரொம்ப slow ஆக இருக்கு, அதனால் "
    "application logs, database connection, மற்றும் network policy எல்லாத்தையும் "
    "கவனமாக check பண்ணுங்க."
)

# The private 7.127 s prompt begins 1.032 s into the supplied 30 s source,
# after the opening 'தோழர்களே'. The previous secret transcript incorrectly
# added 'வணக்கம்' and repeated that opening word.
REF_TEXT_SHORT = (
    "தோழர் ஸ்டாலின் அவர்கள் தன்னுடைய வேட்புமனு தாக்கல் செய்திருப்பார்ல, "
    "அதுக்குள்ள அவரோட சொத்து விவரங்களையெல்லாம் போட்டிருப்பார்ல."
)
REF_TEXT_EXTENDED = REF_TEXT_SHORT + " அந்த 2006 தேர்தலாருக்கட்டும்,"

CANDIDATES = [
    {
        "id": "a_exact_default",
        "ref_text": REF_TEXT_SHORT,
        "speed": 0.95,
        "cfg_strength": 2.0,
        "nfe_step": 32,
        "sway_sampling_coef": -1.0,
        "seed": 20260816,
    },
    {
        "id": "b_exact_identity_soft",
        "ref_text": REF_TEXT_SHORT,
        "speed": 0.95,
        "cfg_strength": 1.5,
        "nfe_step": 40,
        "sway_sampling_coef": -1.0,
        "seed": 20260816,
    },
    {
        "id": "c_extended_identity",
        "ref_text": REF_TEXT_EXTENDED,
        "speed": 0.95,
        "cfg_strength": 1.7,
        "nfe_step": 40,
        "sway_sampling_coef": -1.0,
        "seed": 20260816,
    },
]

CONFIG = {
    "architectures": ["INF5Model"],
    "auto_map": {
        "AutoConfig": "model.INF5Config",
        "AutoModel": "model.INF5Model",
    },
    "ckpt_path": "checkpoints/model_best.pt",
    "model_type": "inf5",
    "remove_sil": True,
    "speed": 1.0,
    "cfg_strength": 2.0,
    "nfe_step": 32,
    "sway_sampling_coef": -1.0,
    "target_rms": 0.1,
    "cross_fade_duration": 0.15,
    "torch_dtype": "float32",
    "transformers_version": "4.49.0",
    "vocab_path": "checkpoints/vocab.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Dhee-Indic-F5 speaker identity")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", default="outputs/voice_001_identity_tuning")
    return parser.parse_args()


def prepare_snapshot(token: str | None) -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            token=token,
            allow_patterns=["model.safetensors", "checkpoints/*", "f5_tts/*", "f5_tts/**/*"],
        )
    )
    model_py = Path(
        hf_hub_download(repo_id=COMPAT_MODEL_ID, filename="model.py", token=token)
    )
    vocab = Path(
        hf_hub_download(
            repo_id=COMPAT_MODEL_ID,
            filename="checkpoints/vocab.txt",
            token=token,
        )
    )

    local_model = snapshot / "model.py"
    local_vocab = snapshot / "checkpoints" / "vocab.txt"
    local_vocab.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_py, local_model)
    shutil.copyfile(vocab, local_vocab)

    source = local_model.read_text(encoding="utf-8")
    vocab_lookup = (
        'vocab_path = hf_hub_download(config.name_or_path, '
        'filename="checkpoints/vocab.txt")'
    )
    if vocab_lookup not in source:
        raise RuntimeError("Compatible loader vocabulary anchor changed")
    source = source.replace(vocab_lookup, 'vocab_path = os.environ["INDICF5_VOCAB_PATH"]')
    source = source.replace(
        "hf_hub_download(config.name_or_path,",
        "hf_hub_download(os.environ.get('INDICF5_WEIGHTS_REPO', config.name_or_path),",
    )

    inference_anchor = """            speed=self.config.speed,\n            device=self.device,"""
    inference_replacement = """            speed=self.config.speed,\n            nfe_step=getattr(self.config, \"nfe_step\", 32),\n            cfg_strength=getattr(self.config, \"cfg_strength\", 2.0),\n            sway_sampling_coef=getattr(self.config, \"sway_sampling_coef\", -1.0),\n            target_rms=getattr(self.config, \"target_rms\", 0.1),\n            cross_fade_duration=getattr(self.config, \"cross_fade_duration\", 0.15),\n            device=self.device,"""
    if inference_anchor not in source:
        raise RuntimeError("Compatible loader inference anchor changed")
    source = source.replace(inference_anchor, inference_replacement)
    local_model.write_text(source, encoding="utf-8")

    (snapshot / "config.json").write_text(
        json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.environ["INDICF5_WEIGHTS_REPO"] = MODEL_ID
    os.environ["INDICF5_VOCAB_PATH"] = str(local_vocab.resolve())
    return snapshot


def normalise_audio(audio: object) -> np.ndarray:
    waveform = np.asarray(audio)
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    waveform = waveform.astype(np.float32).reshape(-1)
    if waveform.size < SAMPLE_RATE // 2 or not np.isfinite(waveform).all():
        raise RuntimeError("Invalid generated waveform")
    if float(np.std(waveform)) < 1e-4 or int(np.unique(np.round(waveform, 6)).size) < 64:
        raise RuntimeError("Generated waveform is silence or DC-only")
    return waveform


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None

    load_started = time.perf_counter()
    snapshot = prepare_snapshot(token)
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    load_seconds = time.perf_counter() - load_started

    results: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        model.config.speed = candidate["speed"]
        model.config.cfg_strength = candidate["cfg_strength"]
        model.config.nfe_step = candidate["nfe_step"]
        model.config.sway_sampling_coef = candidate["sway_sampling_coef"]
        torch.manual_seed(candidate["seed"])
        np.random.seed(candidate["seed"] % (2**32))

        started = time.perf_counter()
        audio = model(
            TARGET_TEXT,
            ref_audio_path=args.reference,
            ref_text=candidate["ref_text"],
        )
        waveform = normalise_audio(audio)
        elapsed = time.perf_counter() - started
        output_path = output_dir / f"{candidate['id']}.wav"
        sf.write(output_path, waveform, SAMPLE_RATE)

        results.append(
            {
                **candidate,
                "target_text": TARGET_TEXT,
                "path": str(output_path),
                "duration_seconds": len(waveform) / SAMPLE_RATE,
                "generation_seconds": elapsed,
                "peak": float(np.max(np.abs(waveform))),
                "ac_rms": float(np.std(waveform)),
                "unique_samples": int(np.unique(np.round(waveform, 6)).size),
            }
        )

    report = {
        "voice_id": "voice_001",
        "model_id": MODEL_ID,
        "compat_model_id": COMPAT_MODEL_ID,
        "model_load_seconds": load_seconds,
        "reference_audio_duration_seconds": 7.127,
        "reference_source_offset_seconds": 1.032,
        "results": results,
        "status": "PENDING_BLIND_HUMAN_COMPARISON",
    }
    (output_dir / "identity_tuning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
