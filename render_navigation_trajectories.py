"""Render every generated navigation trajectory with one Gaussian model load."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from render_trajectory import (
    prepare_trajectory_renderer,
    render_trajectory_with_renderer,
)


def _copy_config(config):
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def discover_trajectories(input_dir, pattern: str) -> list[Path]:
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {input_dir}")
    paths = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No trajectories matched {pattern!r} in {input_dir}"
        )
    return paths


def _write_manifest(path: Path, entries: list[dict], expected_count: int) -> None:
    completed = sum(entry["status"] in {"rendered", "skipped"} for entry in entries)
    failed = sum(entry["status"] == "failed" for entry in entries)
    manifest = {
        "expected_count": expected_count,
        "processed_count": len(entries),
        "completed_count": completed,
        "failed_count": failed,
        "complete": completed == expected_count and failed == 0,
        "trajectories": entries,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def render_batch(config) -> list[dict]:
    trajectories = discover_trajectories(
        config.batch.input_dir, str(config.batch.input_glob)
    )
    expected_count = int(config.batch.expected_count)
    if expected_count > 0 and len(trajectories) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} trajectories but found {len(trajectories)}; "
            "finish trajectory generation or override batch.expected_count"
        )

    output_root = Path(config.batch.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / str(config.batch.manifest_filename)

    print(f"Loading Gaussian model once for {len(trajectories)} trajectories...")
    renderer, reference_cameras = prepare_trajectory_renderer(config)
    entries = []
    for index, trajectory_path in enumerate(trajectories, start=1):
        output_dir = output_root / trajectory_path.stem
        video_path = output_dir / str(config.output.video_name)
        complete_marker = output_dir / ".render_complete"
        if (
            bool(config.batch.skip_completed)
            and complete_marker.is_file()
            and video_path.is_file()
            and video_path.stat().st_size > 0
        ):
            print(f"[{index}/{len(trajectories)}] Skip completed: {trajectory_path.name}")
            entries.append(
                {
                    "trajectory": str(trajectory_path),
                    "output": str(video_path),
                    "status": "skipped",
                }
            )
            _write_manifest(manifest_path, entries, len(trajectories))
            continue

        print(f"\n[{index}/{len(trajectories)}] Render: {trajectory_path.name}")
        render_config = _copy_config(config)
        render_config.trajectory.path = str(trajectory_path)
        render_config.output.directory = str(output_dir)
        try:
            render_trajectory_with_renderer(
                render_config, renderer, reference_cameras
            )
            complete_marker.write_text("complete\n")
            entry = {
                "trajectory": str(trajectory_path),
                "output": str(video_path),
                "status": "rendered",
            }
        except Exception as exc:
            entry = {
                "trajectory": str(trajectory_path),
                "output": str(video_path),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            entries.append(entry)
            _write_manifest(manifest_path, entries, len(trajectories))
            if not bool(config.batch.continue_on_error):
                raise
            print(f"Failed {trajectory_path.name}: {entry['error']}")
            continue
        entries.append(entry)
        _write_manifest(manifest_path, entries, len(trajectories))

    print(f"\nBatch rendering finished. Manifest: {manifest_path}")
    return entries


def load_config():
    parser = argparse.ArgumentParser(
        description="Render a directory of navigation trajectories"
    )
    parser.add_argument(
        "--config", default="configs/church_trajectory_render_100.yaml"
    )
    args, overrides = parser.parse_known_args()
    default_path = Path(__file__).parent / "configs" / "trajectory_render.yaml"
    config = OmegaConf.merge(
        OmegaConf.load(default_path),
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist(overrides),
    )
    OmegaConf.resolve(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return config


if __name__ == "__main__":
    render_batch(load_config())
