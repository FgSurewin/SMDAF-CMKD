import torch
import torch.nn as nn

from models.audio_base.resnet_audio_model import ResNetAudioModel
from models.image_base.image_model import ImageModel

from models.lightning_module.audio_image_kd_base_model import (
    ImageAudioKDCompositeBaseModel,
)

from models.basic_modules.proxy_module import Proxy


class ResnetKDModel(ImageAudioKDCompositeBaseModel):
    def __init__(
        self,
        num_classes,
        label_names,
        kd_config,
        model_size="resnet18",
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes, label_names=label_names, kd_config=kd_config
        )
        self.audio_model = ResNetAudioModel(
            num_classes=num_classes,
            label_names=label_names,
            resnet_type=model_size,
        )
        self.image_model = ImageModel(
            num_classes=num_classes,
            label_names=label_names,
            extractor_type="resnet",
            model_size="resnet50",
        )

        checkpoint_root = "./teacher_checkpoints/"
        checkpint_name = f"resnet50_img_model.ckpt"
        checkpoint_path = checkpoint_root + checkpint_name
        checkpoint = torch.load(checkpoint_path)
        self.image_model.load_state_dict(state_dict=checkpoint["state_dict"])
        print(f"Checkpoint: {checkpoint_path}")
        # self.image_module.eval()
        # Freeze all the parameters in the model
        if not kd_config.get("is_bidirectional", None):
            for param in self.image_model.parameters():
                param.requires_grad = False

        self.save_hyperparameters()

    def parse_batch(self, batch):
        audio_data, image_data, labels = batch
        return {"audio_data": audio_data, "image_data": image_data, "labels": labels}

    def forward(self, data):
        audio_data = data["audio_data"]
        image_data = data["image_data"]
        labels = data["labels"]
        # Audio part
        student_logits = self.audio_model(audio_data)

        # Image part
        teacher_logits = self.image_model(image_data)

        return {
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "labels": labels,
        }


if __name__ == "__main__":
    NUM_CLASSES = 7
    K_FOLD = 2

    KD_CONFIG = {
        "kd_type": "c2kd",
        "lambda_1": 1,
        "lambda_2": 1,
        "lambda_3": 1,
        "krc_threshold": 0,
        "temperature": 2.0,
        "kd_loss_type": "kl",
    }
    model = ResnetKDModel(
        num_classes=NUM_CLASSES, label_names=[], k_fold=2, kd_config=KD_CONFIG
    )

    # B, C, H, W
    audio_data = torch.rand([10, 2, 64, 173])
    # B, C, H, W
    image_data = torch.rand([10, 3, 224, 224])

    labels = torch.randint(0, 7, (10, 1))

    data = dict(audio_data=audio_data, image_data=image_data, labels=labels)

    results = model(data)
    print(results.keys())
    print(results["student_logits"].shape)
    # print(audio_logits.shape, img_logits.shape)
