import argparse
from pathlib import Path

from omegaconf import OmegaConf

from trajectory.trainer import TrajectoryTrainer


def load_config():
    parser = argparse.ArgumentParser(description="Optimize a camera trajectory")
    parser.add_argument("--config", required=True)
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
    TrajectoryTrainer(load_config()).train()
