from __future__ import annotations

import argparse
from dataclasses import replace

from nvap.analysis.green_benchmark import run_green_denoise_benchmark
from nvap.config.types import PreprocessConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark green-channel denoising quality.")
    parser.add_argument("--input", required=True, help="Dataset root path.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument(
        "--strategy",
        choices=["hybrid_auto", "classical_branch_aware", "bm4d", "noise2void", "legacy_anisotropic"],
        default="hybrid_auto",
        help="Green denoising strategy.",
    )
    args = parser.parse_args()

    cfg = replace(PreprocessConfig(), green_denoise_strategy=args.strategy)
    report = run_green_denoise_benchmark(args.input, args.output, cfg)
    print(f"Benchmark report written: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
