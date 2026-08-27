import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def _sample_positions(rng, lower, upper, minimum_distance, max_attempts):
    for _ in range(max_attempts):
        start = rng.uniform(lower, upper).astype(np.float32)
        goal = rng.uniform(lower, upper).astype(np.float32)
        if np.linalg.norm(goal - start) >= minimum_distance:
            return start, goal
    raise RuntimeError(
        "Could not sample start/goal positions with the requested minimum distance"
    )


def _sample_direction(rng, up):
    while True:
        direction = rng.normal(size=3)
        direction = direction - np.dot(direction, up) * up
        if np.linalg.norm(direction) >= 1e-8:
            return _normalize(direction).astype(np.float32)


def _interpolate_directions(start, goal, alphas):
    directions = (
        (1.0 - alphas[:, None]) * start[None]
        + alphas[:, None] * goal[None]
    )
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        return None
    return (directions / norms).astype(np.float32)


def _sample_directions(
    rng,
    up,
    alphas,
    minimum_direction_dot,
    max_attempts,
):
    for _ in range(max_attempts):
        start = _sample_direction(rng, up)
        goal = _sample_direction(rng, up)
        if float(np.dot(start, goal)) < minimum_direction_dot:
            continue
        directions = _interpolate_directions(start, goal, alphas)
        if directions is None:
            continue
        return start, goal, directions
    raise RuntimeError(
        "Could not sample start/goal directions satisfying the up constraints"
    )


def generate_trajectory(config):
    if config.num_waypoints < 2:
        raise ValueError("num_waypoints must be at least 2")
    if config.sampling.max_attempts < 1:
        raise ValueError("sampling.max_attempts must be positive")
    if config.sampling.minimum_endpoint_distance < 0:
        raise ValueError("minimum_endpoint_distance must be non-negative")
    if not -1.0 <= config.sampling.minimum_direction_dot <= 1.0:
        raise ValueError("minimum_direction_dot must be in [-1, 1]")

    lower = np.asarray(config.sampling.position_min, dtype=np.float32)
    upper = np.asarray(config.sampling.position_max, dtype=np.float32)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("position_min and position_max must define a valid 3D box")

    up = np.asarray(config.up, dtype=np.float32)
    if up.shape != (3,) or not np.isfinite(up).all():
        raise ValueError("up must be a finite 3D vector")
    up = _normalize(up).astype(np.float32)

    rng = np.random.default_rng(config.seed)
    start_position, goal_position = _sample_positions(
        rng,
        lower,
        upper,
        config.sampling.minimum_endpoint_distance,
        config.sampling.max_attempts,
    )
    alphas = np.linspace(0.0, 1.0, config.num_waypoints, dtype=np.float32)
    positions = (
        (1.0 - alphas[:, None]) * start_position[None]
        + alphas[:, None] * goal_position[None]
    ).astype(np.float32)

    start_direction, goal_direction, directions = _sample_directions(
        rng,
        up,
        alphas,
        config.sampling.minimum_direction_dot,
        config.sampling.max_attempts,
    )

    output_path = Path(config.output.path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **{
            config.output.position_key: positions,
            config.output.direction_key: directions,
            "start_position": start_position,
            "goal_position": goal_position,
            "start_direction": start_direction,
            "goal_direction": goal_direction,
            "up": up,
            "waypoint_up": np.broadcast_to(up, directions.shape).copy(),
        },
    )
    OmegaConf.save(config, output_path.with_suffix(".yaml"))

    print(f"Saved trajectory: {output_path.resolve()}")
    print(f"start position:  {start_position.tolist()}")
    print(f"goal position:   {goal_position.tolist()}")
    print(f"start direction: {start_direction.tolist()}")
    print(f"goal direction:  {goal_direction.tolist()}")


def load_config():
    parser = argparse.ArgumentParser(
        description="Generate a random linear camera trajectory"
    )
    parser.add_argument("--config", default="configs/trajectory_generate.yaml")
    args, overrides = parser.parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist(overrides),
    )
    OmegaConf.resolve(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return config


if __name__ == "__main__":
    generate_trajectory(load_config())
