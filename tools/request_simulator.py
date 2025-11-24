import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# metadata_path = "./tools/metadata.parquet"
metadata_path = "./tools/filtered_metadata.parquet" # 500 requests
_DEFAULT_PLOT_PATH = "./tools/timestamp_distribution.png"
_DEFAULT_API_URL = os.environ.get("XDIT_SERVER_URL", "http://127.0.0.1:6000/generate")
_DEFAULT_SAVE_PATH = os.environ.get(
    "XDIT_SAVE_DIR", str(Path("output") / "wan" / "sp")
)
_BASE_PAYLOAD = {
    "prompt": "colored sketch in the style of ck-ccd, young Asian woman in a denim jacket, short ruffled skirt, fishnet tights, black high heel boots, sitting in front of a stone wall covered in anime-themed graffiti, long loose hair, outdoors",
    "num_inference_steps": 50,
    "cfg": 7.5,
    "height": 1024,
    "width": 1024,
    "seed": 42,
    "save_disk_path": _DEFAULT_SAVE_PATH,
}

_SAMPLER_DICT = {
    1: "ddim",
    2: "plms",
    3: "k_euler",
    4: "k_euler_ancestral",
    5: "ddik_heunm",
    6: "k_dpm_2",
    7: "k_dpm_2_ancestral",
    8: "k_lms",
    9: "others",
}

def _load_metadata():
    return pd.read_parquet(metadata_path)

def _generate_examples(metadata_df):
    for row in metadata_df.itertuples(index=False):
        timestamp = None if pd.isnull(row.timestamp) else row.timestamp
        yield row.image_name, {
            "image_name": row.image_name,
            "prompt": row.prompt,
            "part_id": row.part_id,
            "seed": row.seed,
            "step": row.step,
            "cfg": row.cfg,
            "sampler": _SAMPLER_DICT[int(row.sampler)],
            "width": row.width,
            "height": row.height,
            "user_name": row.user_name,
            "timestamp": timestamp,
            "image_nsfw": row.image_nsfw,
            "prompt_nsfw": row.prompt_nsfw,
        }

def _analyze_timestamp_distribution(metadata_df, plot_path=_DEFAULT_PLOT_PATH, histogram_bins=48):
    timestamps = metadata_df["timestamp"].dropna().sort_values()
    if timestamps.empty:
        print("No timestamp data available for analysis.")
        return {}

    ts = timestamps
    tz = getattr(ts.dt, "tz", None)
    if tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    numeric_ts = mdates.date2num(ts.tolist())

    start, end = timestamps.iloc[0], timestamps.iloc[-1]
    total_duration = end - start
    deltas = timestamps.diff().dropna()
    hourly_density = timestamps.dt.floor("H").value_counts().sort_index()
    hourly_index = hourly_density.index
    if getattr(hourly_index, "tz", None) is not None:
        hourly_index = hourly_index.tz_convert("UTC").tz_localize(None)

    summary = {
        "count": len(timestamps),
        "range_start": start,
        "range_end": end,
        "total_duration": total_duration,
        "avg_interval": deltas.mean(),
        "median_interval": deltas.median(),
        "min_interval": deltas.min(),
        "max_interval": deltas.max(),
        "peak_hour_count": int(hourly_density.max()),
    }

    print(
        f"Timestamps span from {summary['range_start']} to {summary['range_end']} "
        f"({summary['total_duration']})."
    )
    print(
        f"Median interval: {summary['median_interval']}, "
        f"Average interval: {summary['avg_interval']}."
    )
    print(
        f"Min interval: {summary['min_interval']} | "
        f"Max interval: {summary['max_interval']}."
    )
    print(f"Peak per-hour density: {summary['peak_hour_count']} samples.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].hist(numeric_ts, bins=histogram_bins, color="#5DA5DA", alpha=0.8)
    axes[0].set_title("Timestamp Distribution")
    axes[0].set_ylabel("Count")
    axes[0].xaxis_date()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    axes[1].plot(hourly_index, hourly_density.values, color="#F17CB0")
    axes[1].set_title("Per-Hour Generation Density")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Count")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()

    if plot_path:
        fig.savefig(plot_path, dpi=150)
        print(f"Saved timestamp distribution plot to {plot_path}")
    else:
        plt.show()
    plt.close(fig)
    return summary

class RequestSimulator:
    def __init__(self, plot_path=_DEFAULT_PLOT_PATH):
        self.metadata_df = _load_metadata()
        self.meta_datas = list(_generate_examples(self.metadata_df))
        print(f"Loaded {len(self.meta_datas)} metadata entries.")
        # Prepare ordered entries and streaming state for chunked reading
        self._entries = [meta for _, meta in self.meta_datas if meta.get("timestamp") is not None]
        self._entries.sort(key=lambda m: m["timestamp"])  # ascending by timestamp
        self._pos = 0
        self._prev_ts = None
        # self.timestamp_analysis = _analyze_timestamp_distribution(
        #     self.metadata_df, plot_path=plot_path
        # )

    def _build_payload(self, meta):
        """Merge metadata fields into the default payload."""
        payload = dict(_BASE_PAYLOAD)
        if meta.get("prompt") is not None:
            payload["prompt"] = meta["prompt"]
        if meta.get("step") is not None and pd.notnull(meta["step"]):
            try:
                payload["num_inference_steps"] = int(meta["step"])  # step -> num_inference_steps
            except Exception:
                pass
        if meta.get("cfg") is not None and pd.notnull(meta["cfg"]):
            try:
                payload["cfg"] = float(meta["cfg"])
            except Exception:
                pass
        if meta.get("width") is not None and pd.notnull(meta["width"]):
            try:
                payload["width"] = int(meta["width"])
            except Exception:
                pass
        if meta.get("height") is not None and pd.notnull(meta["height"]):
            try:
                payload["height"] = int(meta["height"])
            except Exception:
                pass
        if meta.get("seed") is not None and pd.notnull(meta["seed"]):
            try:
                payload["seed"] = int(meta["seed"])  # optional seed supported by server
            except Exception:
                pass
        return payload

    def simulate(self, num_requests=10):
        import time
        import requests

        # Filter entries with timestamp and sort chronologically
        entries = [meta for _, meta in self.meta_datas if meta.get("timestamp") is not None]
        if not entries:
            print("No entries with timestamp found; nothing to simulate.")
            return

        entries.sort(key=lambda m: m["timestamp"])  # ascending
        if num_requests is not None and num_requests > 0:
            entries = entries[: num_requests]

        prev_ts = None
        for idx, meta in enumerate(entries):
            ts = meta["timestamp"]
            if prev_ts is not None:
                delta = (ts - prev_ts).total_seconds()
                if delta > 0:
                    time.sleep(delta)
            prev_ts = ts

            payload = self._build_payload(meta)

            try:
                resp = requests.post(_DEFAULT_API_URL, json=payload, timeout=30)
                try:
                    content = resp.json()
                except Exception:
                    content = {"text": resp.text[:200]}
                print(f"[{idx+1}/{len(entries)}] {ts} -> status {resp.status_code}, resp: {content}")
            except Exception as e:
                print(f"[{idx+1}/{len(entries)}] {ts} -> request failed: {e}")

    def simulate_next_chunk(self, chunk_size=100):
        if self._pos >= len(self._entries):
            print("All entries have been processed.")
            return

        import time
        import requests

        start = self._pos
        end = min(self._pos + int(chunk_size), len(self._entries))
        entries = self._entries[start:end]
        self._pos = end

        for idx, meta in enumerate(entries, start=1):
            ts = meta["timestamp"]
            if self._prev_ts is not None:
                delta = (ts - self._prev_ts).total_seconds()
                if delta > 0:
                    time.sleep(delta)
            self._prev_ts = ts

            payload = self._build_payload(meta)

            try:
                resp = requests.post(_DEFAULT_API_URL, json=payload, timeout=30)
                try:
                    content = resp.json()
                except Exception:
                    content = {"text": resp.text[:200]}
                print(
                    f"[chunk {start+idx}/{len(self._entries)}] {ts} -> status {resp.status_code}, resp: {content}"
                )
            except Exception as e:
                print(f"[chunk {start+idx}/{len(self._entries)}] {ts} -> request failed: {e}")

    def filter_and_save(self, file, req_num, start_index, min_time_interval):
        """
        Filter entries by timestamp spacing and persist the result in metadata-compatible format.

        Args:
            file (str): Destination path. Use a `.parquet` extension to match the original metadata schema.
            req_num (int): Maximum number of requests to keep after filtering. If falsy, keep all.
            start_index (int): 0-based index into the sorted entries to start from.
            min_time_interval (float): Minimum number of seconds allowed between two requests.
        """
        timestamped_df = (
            self.metadata_df[self.metadata_df["timestamp"].notnull()]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if timestamped_df.empty:
            print("No timestamped entries available; nothing to save.")
            return

        if start_index < 0:
            start_index = 0
        if start_index >= len(timestamped_df):
            print(f"start_index {start_index} is out of range (entries: {len(timestamped_df)}).")
            return
        if min_time_interval is None or min_time_interval < 0:
            raise ValueError("min_time_interval must be a non-negative number of seconds.")

        selected_indices = []
        prev_ts = None
        subset_df = timestamped_df.iloc[start_index:]
        for idx, row in subset_df.iterrows():
            ts = row["timestamp"]
            if prev_ts is not None:
                delta = (ts - prev_ts).total_seconds()
                if delta < min_time_interval:
                    continue
            selected_indices.append(idx)
            prev_ts = ts
            if req_num and len(selected_indices) >= req_num:
                break

        if not selected_indices:
            print("No entries satisfied the filtering rules; nothing saved.")
            return

        filtered_df = timestamped_df.loc[selected_indices].reset_index(drop=True)
        suffix = Path(file).suffix.lower()
        if suffix == ".parquet":
            filtered_df.to_parquet(file, index=False)
            saved_format = "Parquet"
        elif suffix in {".jsonl", ".json"}:
            filtered_df.to_json(file, orient="records", lines=True, date_format="iso")
            saved_format = "JSONL"
        else:
            filtered_df.to_parquet(file, index=False)
            saved_format = "Parquet"

        print(f"Saved {len(filtered_df)} filtered requests to {file} as {saved_format}.")


if __name__ == "__main__":
    simulator = RequestSimulator()
