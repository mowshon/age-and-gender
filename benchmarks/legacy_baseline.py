#!/usr/bin/env python3
"""Extract a machine-readable baseline from one or more oracle reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    manifests = [
        json.loads((path.parent / "reference-manifest.json").read_text(encoding="utf-8"))
        for path in args.reports
    ]
    result = {
        "schema_version": 1,
        "measurement_note": (
            "model/input setup and instrumented export are reported separately; "
            "cold/warm totals retain legacy per-call RGB copies and subnet copies"
        ),
        "workloads": [
            {
                "report": str(path),
                "input": manifest["input"],
                "models": manifest["models"],
                "machine": manifest["machine"],
                "oracle": manifest["oracle"],
                "model_load_ms": report["timing_ms"]["model_load"],
                "input_load_ms": report["timing_ms"]["input_load"],
                "cold_total_ms": report["timing_ms"]["cold_total"],
                "cold_input_copy_ms": report["timing_ms"]["cold_input_copy"],
                "cold_frontend_ms": report["timing_ms"]["cold_frontend"],
                "cold_cnn_ms": report["timing_ms"]["cold_cnn"],
                "cold_subnet_copy_ms": report["timing_ms"]["cold_subnet_copy"],
                "warm_total_ms": report["timing_ms"]["warm_total"],
                "warm_input_copy_ms": report["timing_ms"]["warm_input_copy"],
                "warm_subnet_copy_ms": report["timing_ms"]["warm_subnet_copy"],
                "warm_frontend_ms": report["timing_ms"]["warm_frontend"],
                "warm_cnn_ms": report["timing_ms"]["warm_cnn"],
                "instrumented_export_ms": report["timing_ms"]["instrumented_export"],
                "inference_peak_rss_kib": report["inference_peak_rss_kib"],
                "process_peak_rss_kib": report["process_peak_rss_kib"],
                "warm_total_ms_summary": {
                    "count": len(report["timing_ms"]["warm_total"]),
                    "minimum": min(report["timing_ms"]["warm_total"]),
                    "median": statistics.median(report["timing_ms"]["warm_total"]),
                    "maximum": max(report["timing_ms"]["warm_total"]),
                },
            }
            for path, report, manifest in zip(args.reports, reports, manifests)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
