from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DatasetConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataModuleConfig:
    metadata_path: str
    batch_size: int
    k_fold: int
    num_workers: int
    random_state: int
    is_val_from_train: bool


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int
    dev_run: bool
    num_classes: int
    needed_labels_file_path: str
    save_checkpoint_to_cloud: bool
    bucket_name: str
    creds: str
    early_stop_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Optimization:
    optimizer: str
    lr_scheduler: str
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler_config: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler_monitor_metric: str = None


@dataclass
class WandbConfig:
    wandb_project: str
    wandb_mode: str
    log_model: bool


@dataclass
class CheckpointConfig:
    checkpoint_root_path: str
    checkpoint: str
    multimodal: bool


@dataclass
class Config:
    dataset: DatasetConfig
    data_module: DataModuleConfig
    model: ModelConfig
    train: TrainConfig
    optimization: Optimization
    wandb: WandbConfig
    checkpoint_config: CheckpointConfig
