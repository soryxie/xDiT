from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def read_request_records(path: Path) -> List[Dict[str, Any]]:
    """Load request records from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} of {path}") from exc

    if not records:
        raise ValueError(f"No records could be read from {path}")
    return records


def ensure_output_path(path: Path) -> None:
    """Create parent directories for the output file if needed."""
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def filter_records(
    records: Sequence[Dict[str, Any]], required: Iterable[str]
) -> Tuple[List[Dict[str, Any]], int]:
    """Keep only records that contain all required keys (non-None)."""
    required_set = set(required)
    kept: List[Dict[str, Any]] = []
    skipped = 0
    for r in records:
        if required_set.issubset(k for k, v in r.items() if v is not None):
            kept.append(r)
        else:
            skipped += 1
    return kept, skipped


def series_for_load(records: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[int], List[int]]:
    """Build stepwise time series for running and queued request counts."""
    if not records:
        raise ValueError("No records provided for load timeline.")

    events = []
    for r in records:
        arrival = r.get("arrival_time")
        start = r.get("exec_start_time")
        done = r.get("completion_time")
        if arrival is None or start is None or done is None:
            continue
        events.append(("arrival", float(arrival)))
        events.append(("start", float(start)))
        events.append(("finish", float(done)))

    if not events:
        raise ValueError("No complete timing events available for load timeline.")

    order = {"arrival": 0, "start": 1, "finish": 2}
    events.sort(key=lambda item: (item[1], order[item[0]]))
    origin = min(t for _, t in events)

    times: List[float] = [0.0]
    running: List[int] = [0]
    queued: List[int] = [0]

    cur_running = 0
    cur_queued = 0
    last_time = origin

    for kind, event_time in events:
        if event_time != last_time:
            times.append(event_time - origin)
            running.append(cur_running)
            queued.append(cur_queued)

        if kind == "arrival":
            cur_queued += 1
        elif kind == "start":
            cur_queued = max(cur_queued - 1, 0)
            cur_running += 1
        elif kind == "finish":
            cur_running = max(cur_running - 1, 0)

        times.append(event_time - origin)
        running.append(cur_running)
        queued.append(cur_queued)
        last_time = event_time

    return times, running, queued


def binned_throughput(
    records: Sequence[Dict[str, Any]], bin_seconds: float
) -> Tuple[List[float], List[float]]:
    """Compute throughput (completions/sec) per time bin."""
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive.")
    if not records:
        raise ValueError("No records provided for throughput calculation.")

    arrivals = [float(r["arrival_time"]) for r in records]
    completions = [float(r["completion_time"]) for r in records]
    origin = min(arrivals)
    end_time = max(completions)
    total_bins = int((end_time - origin) // bin_seconds) + 1

    bins = [0] * (total_bins + 1)
    for completion in completions:
        idx = int((completion - origin) // bin_seconds)
        bins[idx] += 1

    times = [i * bin_seconds for i in range(len(bins))]
    values = [count / bin_seconds for count in bins]
    return times, values


def basic_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Compute queue wait, exec time, end-to-end latency, and throughput."""
    if not records:
        raise ValueError("No records provided for metrics calculation.")

    arrivals = [float(r["arrival_time"]) for r in records]
    completions = [float(r["completion_time"]) for r in records]
    starts = [float(r["exec_start_time"]) for r in records]

    wall_duration = max(completions) - min(arrivals)
    queue_waits = [s - a for a, s in zip(arrivals, starts)]
    exec_times = [c - s for c, s in zip(completions, starts)]
    end_to_end = [c - a for a, c in zip(arrivals, completions)]

    throughput = len(records) / wall_duration if wall_duration > 0 else 0.0
    return {
        "wall_duration": wall_duration,
        "average_queue_wait": sum(queue_waits) / len(queue_waits),
        "average_exec_time": sum(exec_times) / len(exec_times),
        "average_end_to_end": sum(end_to_end) / len(end_to_end),
        "throughput": throughput,
    }
