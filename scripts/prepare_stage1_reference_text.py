from __future__ import annotations

import argparse
from pathlib import Path

from faster_whisper import WhisperModel

from run_indicf5_diagnostic import resolve_reference_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Privately resolve the Stage 1 reference transcript")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-text-file", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = Path(args.reference)
    protected_text = Path(args.reference_text_file).read_text(encoding="utf-8").strip()
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    verifier = WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )
    alignment, effective_ref_text = resolve_reference_text(verifier, reference, protected_text)
    if not bool(alignment["accepted"]):
        raise SystemExit("Reference transcript alignment/consensus gate failed")
    if not effective_ref_text.strip():
        raise SystemExit("Resolved reference transcript is empty")

    output_file.write_text(effective_ref_text.strip(), encoding="utf-8")
    output_file.chmod(0o600)
    print("reference_text_preparation=PASS")
    print(f"reference_text_source={alignment['mode']}")


if __name__ == "__main__":
    main()
