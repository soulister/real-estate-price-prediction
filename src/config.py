import yaml
from pathlib import Path

from src.paths import PROJECT_ROOT, resolve


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: str | None = None) -> dict:
    path = resolve(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(config: dict, *keys: str) -> Path:
    value = config
    for key in keys:
        value = value[key]
    return resolve(value)
