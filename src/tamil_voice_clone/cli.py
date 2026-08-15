import argparse
from pathlib import Path

from .assets import CAMPPLUS_VOXCELEB, download_asset
from .audio import inspect_reference
from .cache import save_voice_cache
from .phonemes import EspeakPhonemizer, split_language_spans, tagged_phoneme_text
from .speaker import SherpaOnnxSpeakerEncoder, SpeakerEncoderConfig
from .text import normalize_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tvc")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = sub.add_parser("inspect-reference", help="Validate reference audio")
    inspect_cmd.add_argument("path")

    normalize_cmd = sub.add_parser("normalize", help="Normalize Tamil/English/Tanglish text")
    normalize_cmd.add_argument("text")

    spans_cmd = sub.add_parser(
        "language-spans",
        help="Show Tamil/English spans before phonemization",
    )
    spans_cmd.add_argument("text")

    phonemize_cmd = sub.add_parser(
        "phonemize",
        help="Convert Tamil/English/Tanglish text to language-tagged IPA with eSpeak-NG",
    )
    phonemize_cmd.add_argument("text")
    phonemize_cmd.add_argument("--espeak", default="espeak-ng")

    download_cmd = sub.add_parser(
        "download-speaker-model",
        help="Download the lightweight ONNX speaker encoder",
    )
    download_cmd.add_argument("--dir", default="models", dest="directory")

    encode_cmd = sub.add_parser(
        "encode-speaker",
        help="Create a reusable zero-shot speaker cache from reference speech",
    )
    encode_cmd.add_argument("reference")
    encode_cmd.add_argument("--model", required=True)
    encode_cmd.add_argument("--name", default="voice")
    encode_cmd.add_argument("--output", required=True)
    encode_cmd.add_argument("--threads", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "inspect-reference":
        info = inspect_reference(args.path)
        print(
            f"ok duration={info.duration_seconds:.1f}s "
            f"sample_rate={info.sample_rate} channels={info.channels}"
        )
        return

    if args.command == "normalize":
        profile = normalize_text(args.text)
        print(profile.normalized)
        print(
            f"tamil={profile.has_tamil} latin={profile.has_latin} "
            f"code_mixed={profile.is_code_mixed}"
        )
        return

    if args.command == "language-spans":
        for span in split_language_spans(args.text):
            print(f"{span.language}\t{span.text}")
        return

    if args.command == "phonemize":
        phonemizer = EspeakPhonemizer(executable=args.espeak)
        spans = phonemizer.phonemize(args.text)
        for span in spans:
            print(f"{span.language}\t{span.source_text}\t{span.ipa}")
        print(tagged_phoneme_text(spans))
        return

    if args.command == "download-speaker-model":
        path = download_asset(CAMPPLUS_VOXCELEB, Path(args.directory))
        print(path)
        return

    if args.command == "encode-speaker":
        reference = Path(args.reference)
        inspect_reference(reference)
        encoder = SherpaOnnxSpeakerEncoder(
            SpeakerEncoderConfig(
                model_path=Path(args.model),
                num_threads=args.threads,
            )
        )
        condition = encoder.encode_speaker(reference)
        info = save_voice_cache(Path(args.output), args.name, condition)
        print(
            f"saved={args.output} name={info.name} "
            f"seconds={info.source_seconds:.1f} dims={info.embedding_size}"
        )
