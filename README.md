# tamil-voice-clone

Lightweight zero-shot Tamil + English + Tanglish voice cloning research project.

> **Stage 1: COMPLETE.** The cloned bilingual voice has been human-accepted at approximately 92% similarity for this milestone. Minor Tamil↔English transition smoothing is intentionally deferred. The accepted synthesis path is validated on Apple Silicon under the revised hard 3 GiB RSS ceiling.

## Stage 1 validated baseline

- Model: `dheeyantra/dhee-indic-f5` with the validated IndicF5 compatibility path.
- One model/speaker identity is used for Tamil, English, and mixed Tamil+English speech.
- Effective EMA and vocoder weights are structurally and checksum verified before synthesis.
- Mixed-language synthesis keeps the accepted same-model span strategy with a 55 ms crossfade.
- No paid TTS API is required; the Stage 1 solution is intended for local execution.

### Apple Silicon validation

GitHub Actions workflow: `voice-001-stage1-apple-silicon`

Final validation run: **#8 / run ID `31976534550`**

- Runner architecture: `arm64`
- Runner memory: 7.0 GiB
- TTS-only peak RSS: **2,182,889,472 bytes / 2.033 GiB**
- Hard Stage 1 memory ceiling: **3.0 GiB**
- Memory gate: **PASS**
- Effective model weights: **verified**
- English strict ASR similarity: **1.0**
- English decode consensus: **1.0**
- Mixed strict ASR similarity: **1.0**
- Mixed decode consensus: **1.0**
- Mixed Tamil ASR similarity: **1.0**
- Required English keyword coverage: **1.0**

Selected runtime measurements from the same run:

- English seed 11: 165.516 s generation for 5.312 s audio, RTF **31.159**.
- Mixed seed 31: 396.839 s generation for 5.287 s audio, RTF **75.055**.

These timings are **not** the audiobook performance target. Stage 1 was optimized for reliable cloned-voice quality and memory validation, not long-form generation speed.

## Stage 2

The lightweight audiobook engine will be developed in a **separate repository** so the accepted Stage 1 voice baseline remains stable.

Stage 2 minimum performance target on an Apple M2 Mac with 8 GB unified memory:

- **1 minute of finished audiobook audio must generate in no more than 2 minutes** (`RTF <= 2.0`).
- Preserve the accepted Stage 1 speaker identity and Tamil/English/Tanglish quality.
- Local-only runtime.
- Free/open-source components only.

Stage 1 should not be modified for audiobook-speed experiments unless a Stage 2 change must be backported to correct a genuine cloning defect.
