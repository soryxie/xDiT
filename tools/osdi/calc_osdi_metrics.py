from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .data_utils import basic_metrics, ensure_output_path, filter_records, read_request_records
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from data_utils import basic_metrics, ensure_output_path, filter_records, read_request_records  # type: ignore


def format_metrics(metrics: dict) -> str:
    return (
        f"wall_duration_seconds: {metrics['wall_duration']:.3f}\n"
        f"average_queue_wait_seconds: {metrics['average_queue_wait']:.3f}\n"
        f"average_exec_time_seconds: {metrics['average_exec_time']:.3f}\n"
        f"average_end_to_end_latency_seconds: {metrics['average_end_to_end']:.3f}\n"
        f"throughput_requests_per_sec: {metrics['throughput']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate data metrics from request JSONL.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("output/wan/sp/request-exec-info-20251124-150927.jsonl"),
        help="Input JSONL file with request execution records.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("paper-figures/metrics.txt"),
        help="Output text file for metrics summary.",
    )
    args = parser.parse_args()

    records = read_request_records(args.input)
    filtered, skipped = filter_records(records, ["arrival_time", "exec_start_time", "completion_time"])
    if skipped:
        print(f"Skipped {skipped} records missing required timing fields.")
    if not filtered:
        raise SystemExit("No records with complete timing data were found.")

    metrics = basic_metrics(filtered)
    lines = [
        f"input_file: {args.input}",
        f"total_requests: {len(records)}",
        f"used_requests: {len(filtered)}",
        format_metrics(metrics),
    ]

    ensure_output_path(args.output)
    content = "\n".join(lines) + "\n"
    args.output.write_text(content, encoding="utf-8")
    print(content.strip())
    print(f"Saved metrics report to {args.output}")


if __name__ == "__main__":
    main()
