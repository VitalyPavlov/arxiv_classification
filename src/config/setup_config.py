from pathlib import Path
from omegaconf import OmegaConf
from typing import Any

BASE_DIR = Path(__file__).resolve().parent


class Config:
    def __init__(self, version: str = "default"):
        self.cfg = OmegaConf.create()

        self._load_config(BASE_DIR)
        self.cfg.version = version

    def _load_config(self, path: Path):
        for file in [x for x in path.glob("**/*.yaml") if x.is_file()]:
            config_data = OmegaConf.load(file)
            self.cfg = OmegaConf.merge(self.cfg, config_data)

    def __getattr__(self, item: Any):
        return getattr(self.cfg, item)
