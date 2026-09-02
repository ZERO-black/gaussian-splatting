import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from optimize_navigation_trajectories import discover_trajectories
from trajectory.trainer import TrajectoryTrainer


class BatchTrajectoryOptimizationTest(unittest.TestCase):
    def test_discovers_generated_paths_in_filename_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "navigation_0002.npz").touch()
            (root / "navigation_0001.npz").touch()
            (root / "summary.json").touch()

            paths = discover_trajectories(root, "navigation_*.npz")

            self.assertEqual(
                [path.name for path in paths],
                ["navigation_0001.npz", "navigation_0002.npz"],
            )

    def test_nonempty_run_can_resume_its_own_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "navigation_0001"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "latest.pth"
            checkpoint.touch()

            trainer = object.__new__(TrajectoryTrainer)
            trainer.output_dir = run_dir.resolve()
            trainer.checkpoint_dir = checkpoint_dir.resolve()
            trainer.preview_dir = (run_dir / "previews").resolve()
            trainer.config = OmegaConf.create(
                {"checkpoint": {"resume": str(checkpoint)}}
            )

            trainer._prepare_output()

            self.assertTrue(trainer.preview_dir.is_dir())
            self.assertTrue((run_dir / "config.yaml").is_file())

    def test_nonempty_run_rejects_external_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "navigation_0001"
            run_dir.mkdir()
            (run_dir / "partial.txt").touch()
            external_checkpoint = root / "elsewhere.pth"
            external_checkpoint.touch()

            trainer = object.__new__(TrajectoryTrainer)
            trainer.output_dir = run_dir.resolve()
            trainer.checkpoint_dir = (run_dir / "checkpoints").resolve()
            trainer.preview_dir = (run_dir / "previews").resolve()
            trainer.config = OmegaConf.create(
                {"checkpoint": {"resume": str(external_checkpoint)}}
            )

            with self.assertRaisesRegex(FileExistsError, "own checkpoints"):
                trainer._prepare_output()


if __name__ == "__main__":
    unittest.main()
