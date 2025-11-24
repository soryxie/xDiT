from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from .data_utils import ensure_output_path, filter_records, read_request_records, series_for_load
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from data_utils import ensure_output_path, filter_records, read_request_records, series_for_load  # type: ignore


def plot_load(times: List[float], running: List[int], queued: List[int], output_path: Path) -> None:
    stacked = [r + q for r, q in zip(running, queued)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("white")
    ax.fill_between(
        times,
        0,
        running,
        step="post",
        color="#4C9AFF",
        alpha=0.65,
        linewidth=1.5,
        edgecolor="#1B64C4",
        label="Running (in service)",
    )
    ax.fill_between(
        times,
        running,
        stacked,
        step="post",
        color="#FF6B6B",
        alpha=0.55,
        linewidth=1.5,
        edgecolor="#C43F3F",
        label="Queued (waiting start)",
    )
    ax.step(times, stacked, where="post", color="#A83838", linewidth=0.8, alpha=0.6)
    ax.step(times, running, where="post", color="#1B64C4", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Seconds since first arrival")
    ax.set_ylabel("Requests")
    ax.set_title("Server load and queue over time")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()

    ensure_output_path(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot server load/queue timeline from request JSONL.")
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
        default=Path("paper-figures/load-timeline.png"),
        help="Output PNG path for the load timeline figure.",
    )
    args = parser.parse_args()

    records = read_request_records(args.input)
    filtered, skipped = filter_records(records, ["arrival_time", "exec_start_time", "completion_time"])
    if skipped:
        print(f"Skipped {skipped} records missing required timing fields.")
    if not filtered:
        raise SystemExit("No records with complete timing data were found.")

    times, running, queued = series_for_load(filtered)
    plot_load(times, running, queued, args.output)
    print(f"Saved load timeline to {args.output}")


if __name__ == "__main__":
    main()
