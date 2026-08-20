# Contributing to tamil-voice-clone

Thanks for considering a contribution.

This repository is the Stage-1 research baseline for local Tamil, English, and Tanglish voice cloning, with a focus on one consistent speaker identity, Apple Silicon compatibility, low memory use, and reproducible validation.

## Scope

Stage 1 is intentionally kept stable. Contributions should focus on genuine cloning defects, reproducibility, compatibility, documentation, tests, or clearly isolated improvements. Audiobook-speed experiments belong in the separate Stage-2 repository unless a Stage-1 change is required to fix a cloning defect.

## Contribution rules

- Open an issue before substantial behavioral or architectural changes.
- Keep pull requests focused and reviewable.
- Do not commit private voice references, transcripts, generated personal audio, credentials, tokens, or other sensitive data.
- Preserve the local-only, no-paid-API goal unless a change has been discussed first.
- Do not claim improved speaker similarity or pronunciation from automated metrics alone; human listening remains authoritative.
- Include regression tests and Apple-Silicon validation where relevant.

## Pull requests

Please describe:

- the problem being solved;
- the exact behavior changed;
- test and validation evidence;
- memory/performance impact where applicable;
- any model, package, or license implications;
- whether audio quality has been human-reviewed or remains unreviewed.

## Licensing

By contributing, you agree that your contributions may be distributed under the repository's MIT License.
