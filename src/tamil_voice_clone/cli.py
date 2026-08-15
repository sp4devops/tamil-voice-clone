import argparse

from .audio import inspect_reference
from .text import normalize_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tvc")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = sub.add_parser("inspect-reference", help="Validate reference audio")
    inspect_cmd.add_argument("path")

    normalize_cmd = sub.add_parser("normalize", help="Normalize Tamil/English/Tanglish text")
    normalize_cmd.add_argument("text")
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
