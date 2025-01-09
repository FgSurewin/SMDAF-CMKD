import os
import sys
import torch
import wandb
import importlib
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import lightning as L
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score, MetricCollection, ConfusionMatrix
from sklearn.metrics import classification_report
from utils.path_utils import PathUtils


def get_dynamic_class(import_module_name):
    module_path, class_name = import_module_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class


class SingleBaseModel(L.LightningModule):
    def __init__(self, num_classes, label_names, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.label_names = label_names
        self.kwargs = kwargs
        metrics = MetricCollection(
            [
                Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="macro",
                ),
                F1Score(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average="macro",
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.conf_matrix = ConfusionMatrix(
            task="multiclass", num_classes=self.num_classes
        )
        self.test_pred_list = []
        self.test_target_list = []
        self.test_logits_list = []  # List to store logits

    def training_step(self, batch, batch_idx):
        data, labels = batch
        y_hat = self(data)
        loss = F.cross_entropy(y_hat, labels)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        metrics_dict = self.train_metrics(y_hat, labels)
        self.log_dict(
            metrics_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        y_hat = self(data)
        loss = F.cross_entropy(y_hat, labels)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        val_metrics_dict = self.val_metrics(y_hat, labels)
        self.log_dict(
            val_metrics_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        y_hat = self(data)
        self.test_pred_list += y_hat.argmax(dim=1).tolist()
        self.test_target_list += labels.tolist()
        self.test_logits_list.append(y_hat.tolist())
        loss = F.cross_entropy(y_hat, labels)
        self.test_metrics.update(y_hat, labels)
        self.conf_matrix.update(y_hat, labels)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_test_epoch_end(self):
        # Log the test metrics
        metrics_dict = self.test_metrics.compute()
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, logger=True)
        # Crate a folder to save the test results
        # Create model folder if it does not exist
        root_path = "./metrics"
        base_folder_path = os.path.join(root_path, self.hparams.model_base_name)
        output_folder = (
            datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            if self.hparams.model_name is None
            else self.hparams.model_name
        )
        model_folder_path = os.path.join(base_folder_path, output_folder)
        PathUtils.create_dir(model_folder_path, show_message=False)

        # Flatten the test_logits_list to ensure each row is stored correctly
        flattened_logits_list = [
            logit for sublist in self.test_logits_list for logit in sublist
        ]

        # Save the pred and ground truth to csv
        result_df = pd.DataFrame(
            {
                "pred": self.test_pred_list,
                "target": self.test_target_list,
                "logits": flattened_logits_list,
            }
        )
        result_df.to_csv(
            os.path.join(model_folder_path, "test_result.csv"), index=False
        )

        cr_result = classification_report(
            self.test_target_list,
            self.test_pred_list,
            target_names=self.label_names,
            output_dict=True,
            zero_division=0,
        )
        text_report = classification_report(
            self.test_target_list,
            self.test_pred_list,
            target_names=self.label_names,
            output_dict=False,
        )
        print(text_report)
        # Save the classification report to local file
        with open(
            os.path.join(model_folder_path, "classification_report.txt"), "w"
        ) as f:
            f.write(text_report)

        df = (
            pd.DataFrame(cr_result)
            .drop(columns=["accuracy", "macro avg", "weighted avg"])
            .T.reset_index()
            .rename(columns={"index": "name"})
        )
        self.logger.log_table(dataframe=df, key="test")
        cm = wandb.plot.confusion_matrix(
            y_true=self.test_target_list,
            preds=self.test_pred_list,
            class_names=self.label_names,
        )
        self.logger.experiment.log({"test_confusion_matrix": cm})

        # Save confusion matrix locally
        self.conf_matrix.compute()
        fig_, ax_ = self.conf_matrix.plot()
        fig_.savefig(os.path.join(model_folder_path, "confusion_matrix.png"))

        # Reset
        self.test_metrics.reset()
        self.conf_matrix.reset()

    def configure_optimizers(self):
        Optim = get_dynamic_class(self.hparams.optimizer)
        LrScheduler = get_dynamic_class(self.hparams.lr_scheduler)

        optimizer = Optim(self.parameters(), **self.hparams.optimizer_config)
        scheduler = LrScheduler(
            optimizer,
            **self.hparams.lr_scheduler_config,
        )
        scheduler_dict = {"scheduler": scheduler}
        if self.hparams.get("lr_scheduler_monitor_metric", None) is not None:
            scheduler_dict["monitor"] = self.hparams.lr_scheduler_monitor_metric

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
            # "monitor": self.hparams.monitor_metric,
        }
