from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from .data_utils import binned_throughput, ensure_output_path, filter_records, read_request_records
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from data_utils import binned_throughput, ensure_output_path, filter_records, read_request_records  # type: ignore


def plot_throughput(times, values, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(times, values, color="#4C9AFF", linewidth=2.0, marker="o", markersize=4, label="Completions per second")
    ax.set_xlabel("Seconds since first arrival")
    ax.set_ylabel("Throughput (requests/sec)")
    ax.set_title("Throughput over time")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    ensure_output_path(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot throughput time series from request JSONL.")
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
        default=Path("paper-figures/throughput.png"),
        help="Output PNG path for the throughput figure.",
    )
    parser.add_argument(
        "-b",
        "--bin-seconds",
        type=float,
        default=1.0,
        help="Bin size in seconds for throughput aggregation (default: 1s).",
    )
    args = parser.parse_args()

    records = read_request_records(args.input)
    filtered, skipped = filter_records(records, ["arrival_time", "completion_time"])
    if skipped:
        print(f"Skipped {skipped} records missing arrival or completion times.")
    if not filtered:
        raise SystemExit("No records with arrival and completion times were found.")

    times, values = binned_throughput(filtered, args.bin_seconds)
    plot_throughput(times, values, args.output)
    print(f"Saved throughput plot to {args.output}")


if __name__ == "__main__":
    main()
