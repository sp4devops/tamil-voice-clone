from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open

from run_dhee_baseline import prepare_compatible_snapshot, validate_waveform
from run_indicf5_diagnostic import (
    COMPAT_MODEL_ID,
    MODEL_ID,
    load_compat_model_direct,
)
from run_indicf5_stage1_multilingual import (
    SAMPLE_RATE,
    TARGETS,
    synthesize_mixed_spans,
    synthesize_with_unicode_duration,
)

SELECTED_SEEDS = {"english": 11, "mixed": 31}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTS-only Stage 1 IndicF5 runtime benchmark")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--effective-reference-text-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _unwrap(module):
    return module._orig_mod if hasattr(module, "_orig_mod") else module


def _checkpoint_target(key: str) -> tuple[str | None, str | None]:
    for prefix in ("ema_model._orig_mod.", "ema_model."):
        if key.startswith(prefix):
            return "ema", key.removeprefix(prefix)
    for prefix in ("vocoder._orig_mod.", "vocoder."):
        if key.startswith(prefix):
            return "vocoder", key.removeprefix(prefix)
    return None, None


def patch_snapshot_for_streaming_weights(snapshot: Path) -> None:
    """Replace compatibility loader's whole-checkpoint materialization with one-tensor-at-a-time copies.

    The compatibility loader normally calls safetensors.load_file(), keeps the full
    checkpoint resident, then creates separate EMA/vocoder dictionaries while the
    destination model is already allocated. This patch changes only weight loading;
    architecture, weights and inference methods are unchanged.
    """

    model_path = snapshot / "model.py"
    source = model_path.read_text(encoding="utf-8")
    source = source.replace("from safetensors.torch import load_file", "from safetensors import safe_open")

    start_marker = "        state_dict = load_file(safetensors_path, device='cpu')\n"
    end_marker = "        # Use eager backend - disables actual compilation while keeping _orig_mod\n"
    if start_marker not in source or end_marker not in source:
        raise RuntimeError("IndicF5 compatibility loader weight-loading block changed")

    start = source.index(start_marker)
    end = source.index(end_marker, start)
    replacement = '''        # Stream checkpoint tensors into the already-allocated modules.\n        # Re-open safetensors for each tensor so mapped checkpoint pages are\n        # released immediately instead of accumulating alongside model RSS.\n        vocab_path = os.environ["INDICF5_VOCAB_PATH"]\n        with torch.device('cpu'):\n            self.ema_model = load_model(\n                DiT,\n                dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),\n                mel_spec_type="vocos",\n                vocab_file=vocab_path,\n                device='cpu'\n            )\n\n        ema = self.ema_model._orig_mod if hasattr(self.ema_model, '_orig_mod') else self.ema_model\n        vocoder = self.vocoder._orig_mod if hasattr(self.vocoder, '_orig_mod') else self.vocoder\n        ema_runtime = ema.state_dict()\n        vocoder_runtime = vocoder.state_dict()\n        seen_ema = set()\n        seen_vocoder = set()\n        unexpected_keys = []\n\n        with safe_open(safetensors_path, framework="pt", device="cpu") as checkpoint_meta:\n            checkpoint_keys = list(checkpoint_meta.keys())\n\n        for checkpoint_key in checkpoint_keys:\n            runtime_group = None\n            runtime_key = None\n            for prefix in ("ema_model._orig_mod.", "ema_model."):\n                if checkpoint_key.startswith(prefix):\n                    runtime_group = "ema"\n                    runtime_key = checkpoint_key[len(prefix):]\n                    break\n            if runtime_group is None:\n                for prefix in ("vocoder._orig_mod.", "vocoder."):\n                    if checkpoint_key.startswith(prefix):\n                        runtime_group = "vocoder"\n                        runtime_key = checkpoint_key[len(prefix):]\n                        break\n            if runtime_group is None or runtime_key is None:\n                continue\n\n            runtime_state = ema_runtime if runtime_group == "ema" else vocoder_runtime\n            seen = seen_ema if runtime_group == "ema" else seen_vocoder\n            if runtime_key not in runtime_state:\n                unexpected_keys.append(checkpoint_key)\n                continue\n\n            with safe_open(safetensors_path, framework="pt", device="cpu") as checkpoint:\n                tensor = checkpoint.get_tensor(checkpoint_key)\n                target = runtime_state[runtime_key]\n                if tuple(tensor.shape) != tuple(target.shape):\n                    unexpected_keys.append(checkpoint_key)\n                else:\n                    with torch.no_grad():\n                        target.copy_(tensor)\n                    seen.add(runtime_key)\n                del tensor\n\n        missing_keys = [f"ema_model.{key}" for key in ema_runtime if key not in seen_ema]\n        missing_keys.extend(f"vocoder.{key}" for key in vocoder_runtime if key not in seen_vocoder)\n'''
    source = source[:start] + replacement + source[end:]
    model_path.write_text(source, encoding="utf-8")


def _tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def prove_streamed_weights(model, snapshot: Path) -> dict[str, object]:
    """Verify full checkpoint structure and representative exact tensor checksums without a full reload."""

    checkpoint_path = snapshot / "model.safetensors"
    ema_runtime = _unwrap(model.ema_model).state_dict()
    vocoder_runtime = _unwrap(model.vocoder).state_dict()

    mappings: dict[str, dict[str, str]] = {"ema": {}, "vocoder": {}}
    checkpoint_shapes: dict[str, dict[str, tuple[int, ...]]] = {"ema": {}, "vocoder": {}}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for checkpoint_key in handle.keys():
            group, runtime_key = _checkpoint_target(checkpoint_key)
            if group is None or runtime_key is None:
                continue
            mappings[group][runtime_key] = checkpoint_key
            checkpoint_shapes[group][runtime_key] = tuple(handle.get_slice(checkpoint_key).get_shape())

    def validate_group(group: str, runtime: dict[str, torch.Tensor]) -> dict[str, object]:
        mapping = mappings[group]
        shapes = checkpoint_shapes[group]
        missing = sorted(key for key in runtime if key not in mapping)
        unexpected = sorted(key for key in mapping if key not in runtime)
        shape_mismatches = sorted(
            key for key in runtime
            if key in shapes and tuple(runtime[key].shape) != shapes[key]
        )
        common = [
            key for key, value in runtime.items()
            if key in mapping
            and key not in shape_mismatches
            and torch.is_floating_point(value)
            and value.numel() >= 4096
        ]
        chosen = sorted(common, key=lambda key: (-runtime[key].numel(), key))[:3]
        samples = []
        for key in chosen:
            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
                checkpoint_tensor = handle.get_tensor(mapping[key])
                runtime_hash = _tensor_digest(runtime[key])
                checkpoint_hash = _tensor_digest(checkpoint_tensor)
                del checkpoint_tensor
            samples.append(
                {
                    "parameter": key,
                    "runtime_checksum": runtime_hash,
                    "checkpoint_checksum": checkpoint_hash,
                    "match": runtime_hash == checkpoint_hash,
                }
            )
        return {
            "runtime_parameter_count": len(runtime),
            "checkpoint_parameter_count": len(mapping),
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "shape_mismatch_count": len(shape_mismatches),
            "sample_count": len(samples),
            "all_sampled_match": bool(samples) and all(bool(item["match"]) for item in samples),
            "samples": samples,
            "verified": bool(
                not missing
                and not unexpected
                and not shape_mismatches
                and samples
                and all(bool(item["match"]) for item in samples)
            ),
        }

    ema = validate_group("ema", ema_runtime)
    vocoder = validate_group("vocoder", vocoder_runtime)
    metrics = {
        "method": "STREAMED_STRUCTURE_AND_CHECKSUM_PROOF",
        "ema": ema,
        "vocoder": vocoder,
        "effective_weights_verified": bool(ema["verified"] and vocoder["verified"]),
    }
    print("streamed_weight_validation=" + json.dumps(metrics, sort_keys=True))
    return metrics


def main() -> None:
    args = parse_args()
    reference = Path(args.reference)
    effective_ref_text = Path(args.effective_reference_text_file).read_text(encoding="utf-8").strip()
    if not effective_ref_text:
        raise SystemExit("Effective reference transcript is empty")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = prepare_compatible_snapshot(
        MODEL_ID,
        COMPAT_MODEL_ID,
        os.environ.get("HF_TOKEN") or None,
    )
    patch_snapshot_for_streaming_weights(snapshot)

    started = time.perf_counter()
    model = load_compat_model_direct(snapshot)
    model_load_seconds = time.perf_counter() - started

    weights = prove_streamed_weights(model, snapshot)
    if not bool(weights["effective_weights_verified"]):
        raise SystemExit("Effective model weight validation failed")

    targets: dict[str, object] = {}
    transient_wavs: list[Path] = []
    try:
        for target_name in ("english", "mixed"):
            seed = SELECTED_SEEDS[target_name]
            torch.manual_seed(seed)
            np.random.seed(seed)
            target_text = str(TARGETS[target_name]["text"])

            started = time.perf_counter()
            with torch.inference_mode():
                if target_name == "mixed":
                    waveform, duration_plan = synthesize_mixed_spans(
                        model=model,
                        reference=reference,
                        effective_ref_text=effective_ref_text,
                    )
                else:
                    waveform, duration_plan = synthesize_with_unicode_duration(
                        model=model,
                        reference=reference,
                        effective_ref_text=effective_ref_text,
                        target_text=target_text,
                    )
            generation_seconds = time.perf_counter() - started

            signal = validate_waveform(waveform, f"runtime_{target_name}")
            wav_path = output_dir / f".{target_name}_runtime.wav"
            sf.write(wav_path, waveform, SAMPLE_RATE, subtype="PCM_16")
            transient_wavs.append(wav_path)
            duration_seconds = float(sf.info(wav_path).duration)

            targets[target_name] = {
                "seed": seed,
                "generation_seconds": round(generation_seconds, 3),
                "duration_seconds": round(duration_seconds, 3),
                "real_time_factor": round(generation_seconds / duration_seconds, 3) if duration_seconds else None,
                "signal_metrics": signal,
                "duration_plan": duration_plan,
            }
            print(
                "runtime_target_metrics="
                + json.dumps(
                    {
                        "target": target_name,
                        "seed": seed,
                        "generation_seconds": round(generation_seconds, 3),
                        "duration_seconds": round(duration_seconds, 3),
                    },
                    sort_keys=True,
                )
            )

        report = {
            "status": "PASS",
            "strategy": "STAGE1_TTS_ONLY_SELECTED_SEEDS_STREAMED_LOAD",
            "architecture": platform.machine(),
            "model_id": MODEL_ID,
            "model_load_seconds": round(model_load_seconds, 3),
            "effective_weights_verified": True,
            "weight_validation": weights,
            "targets": targets,
            "note": "No Whisper verifier model is instantiated; checkpoint tensors are streamed one at a time into the unchanged model architecture.",
        }
        (output_dir / "runtime_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("runtime_benchmark=PASS")
    finally:
        for wav_path in transient_wavs:
            wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
