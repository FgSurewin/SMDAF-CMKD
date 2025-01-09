import sys
import importlib
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, Any
import lightning as L
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def get_dataset_class(dataset_name):
    module_path, class_name = dataset_name.rsplit(".", 1)
    module = importlib.import_module(f"datasets.{module_path}")
    dataset_class = getattr(module, class_name)
    return dataset_class


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        metadata_path: str,
        dataset_name: str,
        dataset_params: Dict[str, Any],
        batch_size: int = 32,
        k_fold: int = 8,
        num_workers: int = 4,
        random_state: int = 88,
        is_val_from_train: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.metadata_path = Path(metadata_path)
        self.batch_size = batch_size
        self.k_fold = k_fold
        self.dataset_name = dataset_name
        self.dataset_params = dataset_params
        self.num_workers = num_workers
        self.random_state = random_state
        self.is_val_from_train = is_val_from_train
        self.kwargs = kwargs

    def setup(self, stage=None):
        # Read the metadata csv file
        self.metadata_df = pd.read_csv(self.metadata_path)

        # Split the data into train and valid using kfold
        # self.train_df = self.metadata_df[self.metadata_df["k_fold"] != self.k_fold]
        self.train_df = self.metadata_df[self.metadata_df["is_train"] == True]
        # self.train_df = self.metadata_df
        if self.is_val_from_train:
            # self.test_df = self.metadata_df[self.metadata_df["k_fold"] == self.k_fold]
            self.test_df = self.metadata_df[self.metadata_df["is_train"] == False]
        else:
            # self.valid_df = self.metadata_df[self.metadata_df["k_fold"] == self.k_fold]
            self.valid_df = self.metadata_df[self.metadata_df["is_train"] == False]

        # Split the data into valid and test
        if self.is_val_from_train:
            self.train_df, self.valid_df = train_test_split(
                self.train_df,
                test_size=0.2,
                random_state=self.random_state,
                stratify=self.train_df["label_id"],
            )
        else:
            self.valid_df, self.test_df = train_test_split(
                self.valid_df,
                test_size=0.5,
                random_state=self.random_state,
                stratify=self.valid_df["label_id"],
            )
            # self.test_df = self.valid_df.copy()

        # Get the dataset class
        self.dataset_class = get_dataset_class(self.dataset_name)

        if stage == "fit":
            self.train_dataset = self.dataset_class(
                df=self.train_df,
                is_augment=True,
                **self.dataset_params,
                **self.kwargs,
            )
            self.valid_dataset = self.dataset_class(
                df=self.valid_df,
                is_augment=False,
                **self.dataset_params,
                **self.kwargs,
            )

        if stage == "test":
            self.test_dataset = self.dataset_class(
                df=self.test_df,
                is_augment=False,
                **self.dataset_params,
                **self.kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
