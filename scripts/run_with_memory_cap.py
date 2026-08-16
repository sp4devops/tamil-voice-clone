#!/usr/bin/env python3
"""Run a command with process-tree RSS and wall-clock limits."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil


def tree_rss_bytes(root: psutil.Process) -> int:
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def terminate_group(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-mib", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--grace-seconds", type=float, default=10.0)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.limit_mib <= 0:
        parser.error("--limit-mib must be positive")
    if args.timeout_seconds < 0:
        parser.error("--timeout-seconds cannot be negative")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.grace_seconds <= 0:
        parser.error("--grace-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a command is required after --")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    try:
        root = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return_code = process.wait()
        root = None

    peak = 0
    reason: str | None = None
    limit = args.limit_mib * 1024 * 1024

    while process.poll() is None:
        elapsed = time.monotonic() - started
        current = tree_rss_bytes(root) if root is not None else 0
        peak = max(peak, current)

        if current > limit:
            reason = "memory"
            print(
                f"Memory cap exceeded: {current / 1024**2:.1f} MiB > {args.limit_mib} MiB",
                file=sys.stderr,
                flush=True,
            )
            terminate_group(process, args.grace_seconds)
            break
        if args.timeout_seconds and elapsed > args.timeout_seconds:
            reason = "timeout"
            print(
                f"Timeout exceeded: {elapsed:.1f}s > {args.timeout_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            terminate_group(process, args.grace_seconds)
            break
        time.sleep(args.poll_seconds)

    return_code = process.poll()
    if return_code is None:
        return_code = process.wait()

    final_rss = 0
    if root is not None:
        try:
            final_rss = tree_rss_bytes(root)
        except psutil.Error:
            final_rss = 0
    peak = max(peak, final_rss)

    report = {
        "command": command,
        "limit_mib": args.limit_mib,
        "timeout_seconds": args.timeout_seconds,
        "peak_rss_mib": round(peak / 1024**2, 2),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "termination_reason": reason,
        "return_code": return_code,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if reason == "memory":
        return 137
    if reason == "timeout":
        return 124
    return int(return_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
