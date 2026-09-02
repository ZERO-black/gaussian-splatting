"""Generate many distinct navigation trajectories in one process.

This is the batch counterpart of ``generate_navigation_trajectory.py``.  The
reference camera and sparse Gaussian proxy are loaded once, while every path
uses a different seed.  Exact collision checks are performed on a batch of
selected lines so the full PLY is not reopened once per trajectory.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from generate_navigation_trajectory import (
    ViewScenario,
    _validate_config,
    load_gaussian_sample,
    sample_safe_segments,
    save_scenario,
    select_view_scenario,
    verify_segments_against_full_ply,
)
from trajectory.reference_camera import load_reference_cameras


@dataclass
class SeededScenario:
    seed: int
    scenario: ViewScenario


def _copy_config(config):
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def _endpoint_distance(first: ViewScenario, second: ViewScenario) -> float:
    """Return endpoint-pair distance, treating reversed A/B as the same path."""
    first_start = first.segment.positions[0]
    first_end = first.segment.positions[-1]
    second_start = second.segment.positions[0]
    second_end = second.segment.positions[-1]
    same_order = max(
        np.linalg.norm(first_start - second_start),
        np.linalg.norm(first_end - second_end),
    )
    reverse_order = max(
        np.linalg.norm(first_start - second_end),
        np.linalg.norm(first_end - second_start),
    )
    return float(min(same_order, reverse_order))


def _is_distinct(
    scenario: ViewScenario, existing: list[SeededScenario], minimum_distance: float
) -> bool:
    return all(
        _endpoint_distance(scenario, item.scenario) >= minimum_distance
        for item in existing
    )


def _output_path(output_dir: Path, filename_format: str, index: int) -> Path:
    filename = filename_format.format(index=index)
    if Path(filename).name != filename or Path(filename).suffix != ".npz":
        raise ValueError(
            "batch.filename_format must produce a plain filename ending in .npz"
        )
    return output_dir / filename


def _validate_batch_config(config) -> None:
    count = int(config.batch.count)
    batch_size = int(config.batch.exact_verification_batch_size)
    if count < 1:
        raise ValueError("batch.count must be positive")
    if batch_size < 1:
        raise ValueError("batch.exact_verification_batch_size must be positive")
    if int(config.batch.max_candidate_attempts) < count:
        raise ValueError("batch.max_candidate_attempts must be at least batch.count")
    if float(config.batch.minimum_endpoint_pair_distance) < 0:
        raise ValueError("batch.minimum_endpoint_pair_distance cannot be negative")


def _write_summary(
    summary_path: Path, entries: list[dict], attempts: int, config
) -> None:
    summary = {
        "requested_count": int(config.batch.count),
        "saved_count": len(entries),
        "candidate_attempts": attempts,
        "seed_start": int(config.batch.seed_start),
        "complete": len(entries) == int(config.batch.count),
        "trajectories": entries,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def generate_batch(config) -> list[Path]:
    _validate_config(config)
    _validate_batch_config(config)

    output_dir = Path(config.batch.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    count = int(config.batch.count)
    start_index = int(config.batch.start_index)
    output_paths = [
        _output_path(output_dir, str(config.batch.filename_format), start_index + i)
        for i in range(count)
    ]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("batch.filename_format does not produce unique filenames")
    existing = [path for path in output_paths if path.exists()]
    if existing and not bool(config.batch.overwrite):
        examples = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} trajectory files: {examples}"
        )

    cameras = load_reference_cameras(
        config.input.camera_json,
        znear=float(config.view.near),
        zfar=float(config.view.far),
    )
    camera_index = int(config.input.camera_index)
    if not -len(cameras) <= camera_index < len(cameras):
        raise IndexError("input.camera_index is out of range")
    reference = cameras[camera_index]

    # The proxy is intentionally fixed across the whole batch. Only A/B sampling
    # changes with each seed, making scenario scores directly comparable.
    sample = load_gaussian_sample(config)
    accepted: list[SeededScenario] = []
    saved_paths: list[Path] = []
    summary_entries: list[dict] = []
    summary_path = output_dir / str(config.batch.summary_filename)
    next_seed = int(config.batch.seed_start)
    attempts = 0
    maximum_attempts = int(config.batch.max_candidate_attempts)
    verification_batch_size = int(config.batch.exact_verification_batch_size)
    minimum_distance = float(config.batch.minimum_endpoint_pair_distance)

    while len(accepted) < count and attempts < maximum_attempts:
        requested = min(verification_batch_size, count - len(accepted))
        candidates: list[SeededScenario] = []
        while len(candidates) < requested and attempts < maximum_attempts:
            seed = next_seed
            next_seed += 1
            attempts += 1
            candidate_config = _copy_config(config)
            candidate_config.seed = seed
            print(
                f"\nCandidate seed={seed} "
                f"({len(accepted)}/{count} accepted, attempt {attempts}/{maximum_attempts})"
            )
            try:
                segments = sample_safe_segments(sample, reference, candidate_config)
                scenario = select_view_scenario(
                    segments, sample, reference, candidate_config
                )
            except RuntimeError as exc:
                print(f"Rejected seed={seed}: {exc}")
                continue
            combined = accepted + candidates
            if not _is_distinct(scenario, combined, minimum_distance):
                print(f"Rejected seed={seed}: endpoints are too similar")
                continue
            candidates.append(SeededScenario(seed, scenario))

        if not candidates:
            break

        if bool(config.collision.exact_verification):
            verified_segments = verify_segments_against_full_ply(
                [item.scenario.segment for item in candidates], config
            )
            verified_ids = {id(segment) for segment in verified_segments}
            candidates = [
                item
                for item in candidates
                if id(item.scenario.segment) in verified_ids
            ]
        for item in candidates:
            output_path = output_paths[len(accepted)]
            save_config = _copy_config(config)
            save_config.seed = item.seed
            save_config.output.path = str(output_path)
            saved_path = save_scenario(item.scenario, reference, save_config)
            accepted.append(item)
            saved_paths.append(saved_path)
            summary_entries.append(
                {
                    "index": len(saved_paths) + start_index - 1,
                    "seed": item.seed,
                    "path": str(saved_path),
                    "start_position": item.scenario.segment.positions[0].tolist(),
                    "goal_position": item.scenario.segment.positions[-1].tolist(),
                    "scenario_score": item.scenario.score,
                    "minimum_clearance": item.scenario.segment.sampled_clearance,
                }
            )
        _write_summary(summary_path, summary_entries, attempts, config)
        print(f"Batch progress: {len(accepted)}/{count} trajectories accepted")

    if len(accepted) < count:
        raise RuntimeError(
            f"Generated only {len(accepted)}/{count} distinct trajectories after "
            f"{attempts} candidate seeds; increase batch.max_candidate_attempts or "
            "relax the path/view constraints"
        )

    _write_summary(summary_path, summary_entries, attempts, config)
    print(f"\nSaved {len(saved_paths)} trajectories and summary: {summary_path}")
    return saved_paths


def load_config():
    parser = argparse.ArgumentParser(
        description="Generate many distinct safe navigation trajectories"
    )
    parser.add_argument(
        "--config", default="configs/church_trajectory_generate_100.yaml"
    )
    args, overrides = parser.parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides)
    )
    OmegaConf.resolve(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return config


if __name__ == "__main__":
    generate_batch(load_config())
