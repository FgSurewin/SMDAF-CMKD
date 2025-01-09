import torch
import torch.nn as nn
from glob import glob
from torch.nn.functional import relu
import torchvision.models as models
import torch.nn.functional as F
import pandas as pd
from config import Config
from sklearn.metrics import f1_score, accuracy_score
from torchmetrics import Accuracy, F1Score
from models.image_base.image_model import ImageModel
from sklearn.model_selection import train_test_split
import os
import hydra
import pandas as pd
from tqdm import tqdm
import importlib
from torch.utils.data import DataLoader


def get_model_class(model_name):
    module_path, class_name = model_name.rsplit(".", 1)
    module = importlib.import_module(f"models.{module_path}")
    model_class = getattr(module, class_name)
    return model_class


def get_dataset_class(dataset_name):
    module_path, class_name = dataset_name.rsplit(".", 1)
    module = importlib.import_module(f"datasets.{module_path}")
    dataset_class = getattr(module, class_name)
    return dataset_class


def get_checkpoint(ckpt_path, folder_path):
    # Use glob to search for .ckpt files
    ckpt_files = glob(os.path.join(folder_path, "*.ckpt"))
    # print("ckpt_files", ckpt_files)
    if not ckpt_files:
        return None  # No .ckpt files found

    if ckpt_path == "best":
        # Return the checkpoint with a random name (assumed to be the best)
        for file in ckpt_files:
            if "last.ckpt" not in file:
                return file
    elif ckpt_path == "last":
        # Return the last checkpoint
        for file in ckpt_files:
            if "last.ckpt" in file:
                return file
    else:
        raise ValueError(
            "Invalid ckpt_path value. It should be either 'best' or 'last'."
        )


@hydra.main(version_base=None, config_path="conf", config_name="test_config.yaml")
def main(cfg: Config):
    print(cfg.model.name)
    ModelClass = get_model_class(cfg.model.name)

    checkpoint_path = get_checkpoint(
        cfg.checkpoint_config.checkpoint, cfg.checkpoint_config.checkpoint_root_path
    )
    print("ckpt_files:", checkpoint_path)
    model = ModelClass.load_from_checkpoint(checkpoint_path)
    metadata_df = pd.read_csv(cfg.data_module.metadata_path)
    test_df = metadata_df[metadata_df["is_train"] == False]
    valid_df, test_df = train_test_split(
        test_df,
        test_size=0.5,
        random_state=88,
        stratify=test_df["label_id"],
    )
    DatasetClass = get_dataset_class(cfg.dataset.name)
    test_dataset = DatasetClass(df=test_df, is_augment=False, **cfg.dataset.params)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        # pin_memory=True,
        # persistent_workers=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_pred_list = []
    test_target_list = []
    test_logits_list = []
    model.to(device)
    model.eval()
    for idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
        if cfg.checkpoint_config.multimodal:
            # data, _, labels = batch
            # data = data.to(device)
            # labels = labels.to(device)
            # y_hat = model.ast_model(data)
            # ---------------------------------------
            data = model.parse_batch(batch)
            data = {key: value.to(device) for key, value in data.items()}
            logits_dict = model(data)
            y_hat = logits_dict["student_logits"]
            labels = data["labels"]
        else:
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)
            y_hat = model(data)
        test_pred_list += y_hat.argmax(dim=1).tolist()
        test_target_list += labels.tolist()
        test_logits_list.append(y_hat.tolist())

    flattened_logits_list = [logit for sublist in test_logits_list for logit in sublist]

    # Save the pred and ground truth to csv
    new_result_df = pd.DataFrame(
        {
            "pred": test_pred_list,
            "target": test_target_list,
            "logits": flattened_logits_list,
        }
    )

    new_pred = new_result_df["pred"].to_numpy()
    new_target = new_result_df["target"].to_numpy()

    accuracy_micro = Accuracy(
        task="multiclass",
        num_classes=7,
        average="micro",
    )
    accuracy_macro = Accuracy(
        task="multiclass",
        num_classes=7,
        average="macro",
    )
    f1_score_micro = F1Score(
        task="multiclass",
        num_classes=7,
        average="micro",
    )
    f1_score_macro = F1Score(
        task="multiclass",
        num_classes=7,
        average="macro",
    )

    new_pred_tensor = torch.from_numpy(new_pred)
    new_target_tensor = torch.from_numpy(new_target)

    print("-----------------------------------------")
    print(
        f"******** Model: {str(cfg.checkpoint_config.checkpoint_root_path).rsplit('/')[-1]} ********"
    )
    print(f"******** Model: {cfg.checkpoint_config.checkpoint} ********")
    print("Micro accuracy: ", accuracy_micro(new_pred_tensor, new_target_tensor))
    print("Macro accuracy: ", accuracy_macro(new_pred_tensor, new_target_tensor))
    print("Micro f1_score: ", f1_score_micro(new_pred_tensor, new_target_tensor))
    print("Macro f1_score: ", f1_score_macro(new_pred_tensor, new_target_tensor))
    print("-----------------------------------------")


if __name__ == "__main__":
    main()
