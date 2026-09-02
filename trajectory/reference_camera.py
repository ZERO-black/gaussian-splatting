import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np


@dataclass(frozen=True)
class ReferenceCamera:
    """Image-free camera metadata loaded from a Graphdeco camera JSON export."""

    image_width: int
    image_height: int
    FoVx: float
    FoVy: float
    c2w: np.ndarray
    image_name: str
    uid: int
    colmap_id: int
    znear: float = 0.01
    zfar: float = 1000.0

    @property
    def position(self) -> np.ndarray:
        return self.c2w[:3, 3]

    @property
    def forward(self) -> np.ndarray:
        return self.c2w[:3, 2]

    @property
    def up(self) -> np.ndarray:
        # Graphdeco C2W stores the camera-down vector in column one.
        return -self.c2w[:3, 1]


def _focal_to_fov(focal: float, pixels: int) -> float:
    if not math.isfinite(focal) or focal <= 0:
        raise ValueError("Camera focal lengths must be finite and positive")
    return 2.0 * math.atan(pixels / (2.0 * focal))


def load_reference_cameras(
    path: Union[str, Path],
    znear: float = 0.01,
    zfar: float = 1000.0,
) -> List[ReferenceCamera]:
    """Load standalone reference cameras from camera.json/cameras.json."""
    path = Path(path).expanduser().resolve()
    with path.open() as stream:
        entries = json.load(stream)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Reference camera JSON must contain a non-empty list: {path}")
    if not 0 < znear < zfar:
        raise ValueError("Reference camera clipping planes must satisfy 0 < near < far")

    cameras = []
    required = {"width", "height", "position", "rotation", "fx", "fy"}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Camera entry {index} must be an object")
        missing = required - set(entry)
        if missing:
            raise ValueError(f"Camera entry {index} is missing {sorted(missing)}")

        width = int(entry["width"])
        height = int(entry["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Camera entry {index} has an invalid resolution")
        position = np.asarray(entry["position"], dtype=np.float64)
        rotation = np.asarray(entry["rotation"], dtype=np.float64)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError(
                f"Camera entry {index} needs position [3] and rotation [3, 3]"
            )
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError(f"Camera entry {index} contains non-finite pose values")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError(f"Camera entry {index} rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError(f"Camera entry {index} rotation must have determinant +1")

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rotation.astype(np.float32)
        c2w[:3, 3] = position.astype(np.float32)
        uid = int(entry.get("id", index))
        cameras.append(
            ReferenceCamera(
                image_width=width,
                image_height=height,
                FoVx=_focal_to_fov(float(entry["fx"]), width),
                FoVy=_focal_to_fov(float(entry["fy"]), height),
                c2w=c2w,
                image_name=str(entry.get("img_name", f"camera_{uid}")),
                uid=uid,
                colmap_id=uid,
                znear=float(znear),
                zfar=float(zfar),
            )
        )
    return cameras
