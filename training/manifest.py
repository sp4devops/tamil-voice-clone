from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingRecord:
    audio: str
    text: str
    speaker: str
    reference_audio: str


def load_pipe_metadata(path: Path) -> list[TrainingRecord]:
    """Load `audio|text|speaker|reference_audio` metadata.

    `reference_audio` should be a clean clip from the same speaker and should
    normally differ from `audio`; this reduces identity leakage from conditioning
    on the exact target utterance during training.
    """
    records: list[TrainingRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for line_number, row in enumerate(reader, start=1):
            if not row or (row[0].lstrip().startswith("#")):
                continue
            if len(row) != 4:
                raise ValueError(
                    f"{path}:{line_number}: expected 4 pipe-delimited fields, got {len(row)}"
                )
            audio, text, speaker, reference_audio = (value.strip() for value in row)
            if not all((audio, text, speaker, reference_audio)):
                raise ValueError(f"{path}:{line_number}: fields cannot be empty")
            records.append(
                TrainingRecord(
                    audio=audio,
                    text=text,
                    speaker=speaker,
                    reference_audio=reference_audio,
                )
            )
    if not records:
        raise ValueError(f"No training records found in {path}")
    return records


def validate_paths(records: list[TrainingRecord], root: Path) -> None:
    missing: list[str] = []
    for record in records:
        for relative in (record.audio, record.reference_audio):
            if not (root / relative).is_file():
                missing.append(relative)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise FileNotFoundError(f"Missing {len(missing)} audio files: {preview}{suffix}")


def write_jsonl(records: list[TrainingRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_pipe_metadata(args.metadata)
    validate_paths(records, args.root)
    write_jsonl(records, args.output)
    speakers = len({record.speaker for record in records})
    print(f"records={len(records)} speakers={speakers} output={args.output}")


if __name__ == "__main__":
    main()
