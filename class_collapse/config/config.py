from dataclasses import dataclass

from omegaconf import DictConfig

@dataclass
class Config:
    hydra_config: DictConfig
    device: str
    