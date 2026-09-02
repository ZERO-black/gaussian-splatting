"""Optimize a directory of navigation trajectories with one shared 3DGS scene."""

import argparse
import gc
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from trajectory.trainer import TrajectoryTrainer


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
    completed = sum(entry["status"] in {"optimized", "skipped"} for entry in entries)
    failed = sum(entry["status"] == "failed" for entry in entries)
    path.write_text(
        json.dumps(
            {
                "expected_count": expected_count,
                "processed_count": len(entries),
                "completed_count": completed,
                "failed_count": failed,
                "complete": completed == expected_count and failed == 0,
                "trajectories": entries,
            },
            indent=2,
        )
        + "\n"
    )


def optimize_batch(config) -> list[dict]:
    trajectories = discover_trajectories(
        config.batch.input_dir, str(config.batch.input_glob)
    )
    expected_count = int(config.batch.expected_count)
    if expected_count > 0 and len(trajectories) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} trajectories but found {len(trajectories)}; "
            "finish generation or override batch.expected_count"
        )

    output_root = Path(config.batch.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / str(config.batch.manifest_filename)
    final_iteration = int(config.optimization.iterations)
    entries = []
    shared_scene = None

    for index, trajectory_path in enumerate(trajectories, start=1):
        run_name = trajectory_path.stem
        run_dir = output_root / run_name
        completion_marker = run_dir / ".optimization_complete"
        final_trajectory = run_dir / f"trajectory_{final_iteration:06d}.npz"
        if (
            bool(config.batch.skip_completed)
            and completion_marker.is_file()
            and final_trajectory.is_file()
        ):
            print(f"[{index}/{len(trajectories)}] Skip completed: {run_name}")
            entries.append(
                {
                    "trajectory": str(trajectory_path),
                    "optimized_trajectory": str(final_trajectory),
                    "status": "skipped",
                }
            )
            _write_manifest(manifest_path, entries, len(trajectories))
            continue

        run_config = _copy_config(config)
        run_config.trajectory.path = str(trajectory_path)
        run_config.output.directory = str(output_root)
        run_config.output.run_name = run_name
        run_config.logging.wandb.name = run_name

        if run_dir.is_dir() and any(run_dir.iterdir()):
            latest_checkpoint = run_dir / "checkpoints" / "latest.pth"
            if latest_checkpoint.is_file() and bool(config.batch.resume_incomplete):
                run_config.checkpoint.resume = str(latest_checkpoint)
                print(
                    f"[{index}/{len(trajectories)}] Resume {run_name} from "
                    f"{latest_checkpoint}"
                )
            else:
                error = (
                    f"Incomplete output has no resumable checkpoint: {run_dir}"
                )
                entries.append(
                    {
                        "trajectory": str(trajectory_path),
                        "optimized_trajectory": str(final_trajectory),
                        "status": "failed",
                        "error": error,
                    }
                )
                _write_manifest(manifest_path, entries, len(trajectories))
                if not bool(config.batch.continue_on_error):
                    raise RuntimeError(error)
                print(error)
                continue
        else:
            print(f"\n[{index}/{len(trajectories)}] Optimize: {run_name}")

        trainer = None
        try:
            trainer = TrajectoryTrainer(run_config, shared_scene=shared_scene)
            if shared_scene is None:
                shared_scene = trainer.shared_scene_context()
                print("Gaussian/KNN scene cached for the remaining trajectories")
            trainer.train()
            completion_marker.write_text("complete\n")
            entry = {
                "trajectory": str(trajectory_path),
                "optimized_trajectory": str(final_trajectory),
                "status": "optimized",
            }
        except Exception as exc:
            entry = {
                "trajectory": str(trajectory_path),
                "optimized_trajectory": str(final_trajectory),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            entries.append(entry)
            _write_manifest(manifest_path, entries, len(trajectories))
            if not bool(config.batch.continue_on_error):
                raise
            print(f"Failed {run_name}: {entry['error']}")
            continue
        finally:
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

        entries.append(entry)
        _write_manifest(manifest_path, entries, len(trajectories))

    print(f"\nBatch optimization finished. Manifest: {manifest_path}")
    return entries


def load_config():
    parser = argparse.ArgumentParser(
        description="Optimize a directory of navigation trajectories"
    )
    parser.add_argument(
        "--config", default="configs/church_trajectory_optimize_100.yaml"
    )
    args, overrides = parser.parse_known_args()
    default_path = Path(__file__).parent / "configs" / "trajectory_train.yaml"
    config = OmegaConf.merge(
        OmegaConf.load(default_path),
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist(overrides),
    )
    OmegaConf.resolve(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return config


if __name__ == "__main__":
    optimize_batch(load_config())
