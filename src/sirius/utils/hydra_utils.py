from pathlib import Path

from hydra import compose
from hydra.initialize import initialize_config_dir
from omegaconf import DictConfig


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs"


def load_config(
    config_name: str,
    overrides: list[str] = [],
    config_path: Path = DEFAULT_CONFIG_PATH,
    version_base: str = None
) -> DictConfig:
    config_path = str(config_path.absolute().resolve())
    with initialize_config_dir(config_dir=config_path, version_base=version_base):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
