#!/usr/bin/env python3
"""Run a command while enforcing an RSS cap across its process tree."""
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


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-mib", type=int, default=8192)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    process = subprocess.Popen(command, start_new_session=True)
    root = psutil.Process(process.pid)
    peak = 0
    exceeded = False
    limit = args.limit_mib * 1024 * 1024

    while process.poll() is None:
        current = tree_rss_bytes(root)
        peak = max(peak, current)
        if current > limit:
            exceeded = True
            print(
                f"Memory cap exceeded: {current / 1024**2:.1f} MiB > {args.limit_mib} MiB",
                file=sys.stderr,
                flush=True,
            )
            terminate_group(process)
            break
        time.sleep(0.25)

    return_code = process.poll()
    if return_code is None:
        return_code = process.wait()
    report = {
        "command": command,
        "limit_mib": args.limit_mib,
        "peak_rss_mib": round(peak / 1024**2, 2),
        "elapsed_seconds": round(time.time() - started, 2),
        "memory_cap_exceeded": exceeded,
        "return_code": return_code,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 137 if exceeded else int(return_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
