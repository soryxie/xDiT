#!/usr/bin/env python3
"""Generate ready-to-run commands for entrypoints/launch.py."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "wan": {
        "label": "Wan2.1",
        "model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    },
    "flux": {
        "label": "Flux.1-dev",
        "model_path": "black-forest-labs/FLUX.1-dev",
    },
    "sd3": {
        "label": "Stable-Diffusion-3",
        "model_path": "stabilityai/stable-diffusion-3-medium-diffusers",
    },
}


def _build_dp_profile(gpu_count: int) -> List[Tuple[str, int]]:
    if gpu_count < 1:
        raise ValueError("GPU count must be at least 1 for DP mode.")
    return [
        ("data_parallel_degree", gpu_count),
        ("pipefusion_parallel_degree", 1),
        ("ulysses_parallel_degree", 1),
    ]


def _build_sp_profile(gpu_count: int) -> List[Tuple[str, int]]:
    if gpu_count < 1:
        raise ValueError("GPU count must be at least 1 for SP mode.")
    return [
        ("data_parallel_degree", 1),
        ("pipefusion_parallel_degree", 1),
        ("ulysses_parallel_degree", gpu_count),
    ]


def _build_pp_profile(gpu_count: int) -> List[Tuple[str, int]]:
    if gpu_count < 1:
        raise ValueError("GPU count must be at least 1 for PP mode.")
    return [
        ("data_parallel_degree", 1),
        ("pipefusion_parallel_degree", gpu_count),
        ("ulysses_parallel_degree", 1),
    ]


MODE_BUILDERS = {
    "sp": {
        "description": "Sequence parallel via Ulysses attention sharding.",
        "builder": _build_sp_profile,
    },
    "pp": {
        "description": "PipeFusion pipeline parallel across GPUs.",
        "builder": _build_pp_profile,
    },
    "dp": {
        "description": "Data parallel (each GPU runs a copy of the model).",
        "builder": _build_dp_profile,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create launch commands for the xDiT HTTP server."
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_REGISTRY, "all"],
        default="all",
        help="Model alias to generate. Use 'all' to emit every model.",
    )
    parser.add_argument(
        "--mode",
        choices=[*MODE_BUILDERS, "all"],
        default="all",
        help="Parallel mode to generate. Use 'all' to emit every mode.",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=4,
        help="Number of GPUs available on the machine.",
    )
    parser.add_argument(
        "--save-root",
        default="output",
        help="Base directory for --save_disk_path.",
    )
    parser.add_argument(
        "--python-bin",
        default=Path(sys.executable).name,
        help="Python executable name used in emitted commands.",
    )
    return parser.parse_args()


def _iter_selected(items: Dict[str, Dict[str, str]], selected: str) -> Iterable[str]:
    if selected == "all":
        return items.keys()
    return [selected]


def _format_command(
    python_bin: str,
    profile: List[Tuple[str, int]],
    model_key: str,
    mode_key: str,
    gpu_count: int,
    save_root: str,
) -> Tuple[str, Dict[str, int]]:
    save_path = Path(save_root) / model_key / mode_key
    cli_parts: List[str] = [
        python_bin,
        "entrypoints/launch.py",
        "--model_path",
        MODEL_REGISTRY[model_key]["model_path"],
        "--world_size",
        str(gpu_count),
        "--save_disk_path",
        str(save_path),
    ]
    values: Dict[str, int] = {}
    for name, value in profile:
        values[name] = value
        cli_parts.extend([f"--{name}", str(value)])
    return shlex.join(cli_parts), values


def _summarize(values: Dict[str, int]) -> str:
    dp = values.get("data_parallel_degree", 1)
    pp = values.get("pipefusion_parallel_degree", 1)
    uly = values.get("ulysses_parallel_degree", 1)
    sp = uly
    total = dp * sp * pp
    return f"dp={dp} x sp={sp} (ulysses={uly}) x pp={pp} -> {total} GPU(s)"


def main() -> None:
    args = _parse_args()
    models = list(_iter_selected(MODEL_REGISTRY, args.model))
    modes = list(_iter_selected(MODE_BUILDERS, args.mode))
    for model_key in models:
        for mode_key in modes:
            builder = MODE_BUILDERS[mode_key]["builder"]
            profile = builder(args.gpu_count)
            command, values = _format_command(
                args.python_bin,
                profile,
                model_key,
                mode_key,
                args.gpu_count,
                args.save_root,
            )
            description = MODE_BUILDERS[mode_key]["description"]
            summary = _summarize(values)
            label = f"[{MODEL_REGISTRY[model_key]['label']}][{mode_key.upper()}]"
            print(label)
            print(f"  {description}")
            print(f"  {command}")
            print(f"  {summary}")
            print()


if __name__ == "__main__":
    main()
