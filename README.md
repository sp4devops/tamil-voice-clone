# tamil-voice-clone

Lightweight zero-shot Tamil + English + Tanglish voice cloning research project.

## Goal

Provide 20–30+ seconds of clean speech from an arbitrary speaker, extract a reusable speaker condition once, then synthesize Tamil, English, or Tanglish in that speaker's voice without per-speaker training.

## Runtime constraints

- CPU-only production inference
- hard RSS ceiling: 2 GB
- target RSS: <=1.5 GB
- low default thread count to reduce sustained CPU load
- offline inference
- no per-speaker fine-tuning

## Current status

This branch is the project bootstrap. It intentionally does **not** fake voice-cloning output before a real zero-shot backend is integrated.

Implemented:

- reference-audio validation
- Tamil/Latin code-mix detection and normalization
- technical-word pronunciation normalization hooks
- zero-shot speaker/model interface
- runtime memory/thread configuration
- CLI inspection commands
- automated tests and CI

Next milestone:

1. benchmark candidate compact speaker encoders
2. integrate a real zero-shot VITS/YourTTS-style baseline
3. add speaker-embedding cache
4. add Tamil + Indian-English phoneme frontend
5. profile peak RSS and real-time factor
6. export/quantize inference components where quality permits

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Validate a reference recording:

```bash
tvc inspect-reference reference.wav
```

Inspect text normalization:

```bash
tvc normalize 'இன்னிக்கு Kubernetes cluster slow'
```
