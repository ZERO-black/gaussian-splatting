import tempfile
import unittest
from pathlib import Path

from render_navigation_trajectories import discover_trajectories


class BatchTrajectoryRenderingTest(unittest.TestCase):
    def test_discovers_only_matching_trajectories_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "navigation_0010.npz").touch()
            (root / "navigation_0002.npz").touch()
            (root / "navigation_0002.json").touch()

            paths = discover_trajectories(root, "navigation_*.npz")

            self.assertEqual(
                [path.name for path in paths],
                ["navigation_0002.npz", "navigation_0010.npz"],
            )

    def test_missing_directory_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            discover_trajectories("/definitely/not/a/trajectory/directory", "*.npz")


if __name__ == "__main__":
    unittest.main()
