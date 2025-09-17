#!/usr/bin/env python3
"""
Measure memory behavior when loading fast-langdetect models.

Credit: script prepared by github@JackyHe398 (adapted for examples/).

Examples

  # Check lite model without limiting memory
  python examples/memory_usage_check.py --model lite

  # Check full model with a 200 MB limit (should pass on many systems)
  python examples/memory_usage_check.py --model full --limit-mb 200

  # Force fallback or failure by using a tight limit
  python examples/memory_usage_check.py --model full --limit-mb 100

Notes
  - RSS measurement uses ru_maxrss which is OS-dependent (kB on Linux, bytes on macOS).
  - Address space limits rely on resource.RLIMIT_AS (primarily effective on Unix-like systems).
  - For accurate results, run this script from a clean terminal session. Running inside IDEs/REPLs can inflate the
    process peak RSS before the script runs, making ru_maxrss appear very large with ~0 delta.
"""

import argparse
import os
import sys
import time
import platform
import resource
from typing import Optional

try:
    from fast_langdetect import detect
except Exception:  # pragma: no cover
    # Support running from repo root without installation
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from fast_langdetect import detect  # type: ignore


def set_address_space_limit(limit_mb: int | None) -> None:
    if limit_mb is None:
        return
    limit_bytes = int(limit_mb) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def format_ru_maxrss_mb(val: int) -> float:
    """Convert ru_maxrss to MB based on OS semantics.

    - Linux: ru_maxrss is in kilobytes
    - macOS (Darwin): ru_maxrss is in bytes
    - BSDs often follow macOS/bytes; treat non-Linux as bytes by default
    """
    system = platform.system()
    if system == "Linux":
        return val / 1024.0
    # Darwin, FreeBSD, etc.: assume bytes
    return val / (1024.0 * 1024.0)


def current_rss_mb() -> Optional[float]:
    """Return current RSS in MB if available; otherwise None.

    Priority:
      1) psutil (if installed)
      2) /proc/self/status (Linux)
    """
    try:
        import psutil  # type: ignore

        p = psutil.Process()
        return p.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass

    if platform.system() == "Linux":
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        # Example: VmRSS:   123456 kB
                        if len(parts) >= 2:
                            kb = float(parts[1])
                            return kb / 1024.0
        except Exception:
            pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check fast-langdetect memory usage and limits.")
    parser.add_argument("--model", choices=["lite", "full", "auto"], default="auto")
    parser.add_argument("--limit-mb", type=int, default=None, help="Set RLIMIT_AS in MB (Unix-like only)")
    parser.add_argument("--text", default="Hello world", help="Text to detect")
    parser.add_argument("--k", type=int, default=1, help="Top-k predictions")
    args = parser.parse_args()

    set_address_space_limit(args.limit_mb)

    print(f"Model: {args.model}")
    if args.limit_mb is not None:
        print(f"Address space limit: {args.limit_mb} MB")

    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    curr_before = current_rss_mb()
    try:
        res = detect(args.text, model=args.model, k=args.k)
    except MemoryError:
        print("MemoryError: model load or inference exceeded limit.")
        return 2
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    curr_after = current_rss_mb()

    peak_used_mb = max(0.0, format_ru_maxrss_mb(peak_after) - format_ru_maxrss_mb(peak_before))
    peak_mb = format_ru_maxrss_mb(peak_after)

    print(f"Result: {res}")
    print(f"Peak RSS (ru_maxrss): ~{peak_mb:.1f} MB")
    print(f"Approx. peak delta: ~{peak_used_mb:.1f} MB")
    if curr_before is not None and curr_after is not None:
        print(f"Current RSS before: ~{curr_before:.1f} MB; after: ~{curr_after:.1f} MB; delta: ~{(curr_after-curr_before):.1f} MB")
    else:
        print("Current RSS: psutil or /proc not available; showing peak only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
