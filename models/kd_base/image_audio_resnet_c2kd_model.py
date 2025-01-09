import torch
import torch.nn as nn

from models.audio_base.resnet_audio_model import ResNetAudioModel
from models.image_base.image_model import ImageModel

from models.lightning_module.audio_image_kd_base_model import (
    ImageAudioKDCompositeBaseModel,
)

from models.basic_modules.proxy_module import C2KDProxy


class ResnetC2KDModel(ImageAudioKDCompositeBaseModel):
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
        for param in self.image_model.img_extractor.conv1.parameters():
            param.requires_grad = False
        for param in self.image_model.img_extractor.bn1.parameters():
            param.requires_grad = False
        for param in self.image_model.img_extractor.layer1.parameters():
            param.requires_grad = False
        for param in self.image_model.img_extractor.layer2.parameters():
            param.requires_grad = False

        dummy_teacher_feat, _ = self.get_image_features(torch.rand((5, 3, 224, 224)))
        self.proxy_teacher = C2KDProxy(dummy_teacher_feat, num_classes)
        dummy_student_feat, _ = self.get_audio_features(torch.rand((5, 2, 64, 173)))
        self.proxy_student = C2KDProxy(dummy_student_feat, num_classes)
        self.save_hyperparameters()

    def get_image_features(self, image_data):
        # test_data = torch.rand((5, 3, 360, 480))  # dummy data
        x = self.image_model.img_extractor.conv1(image_data)
        x = self.image_model.img_extractor.bn1(x)
        x = self.image_model.img_extractor.relu(x)

        x = self.image_model.img_extractor.layer1(x)
        x = self.image_model.img_extractor.layer2(x)
        x = self.image_model.img_extractor.layer3(x)

        x = self.image_model.img_extractor.layer4(x)

        x = self.image_model.img_extractor.avgpool(x)
        avg_x = x
        x = x.view(x.size(0), -1)
        x = self.image_model.img_extractor.fc(x)
        return avg_x, x

    def get_audio_features(self, audio_data):
        # test_data = torch.rand((5, 3, 360, 480))  # dummy data
        x = self.audio_model.upsample(audio_data)
        x = self.audio_model.extractor.conv1(x)
        x = self.audio_model.extractor.bn1(x)
        x = self.audio_model.extractor.relu(x)

        x = self.audio_model.extractor.layer1(x)
        x = self.audio_model.extractor.layer2(x)
        x = self.audio_model.extractor.layer3(x)

        x = self.audio_model.extractor.layer4(x)

        x = self.audio_model.extractor.avgpool(x)
        avg_x = x
        x = x.view(x.size(0), -1)
        x = self.audio_model.extractor.fc(x)

        return avg_x, x

    def parse_batch(self, batch):
        audio_data, image_data, labels = batch
        return {"audio_data": audio_data, "image_data": image_data, "labels": labels}

    def forward(self, data):
        audio_data = data["audio_data"]
        image_data = data["image_data"]
        labels = data["labels"]
        # Audio part
        # student_logits = self.audio_model(audio_data)
        s_feat, student_logits = self.get_audio_features(audio_data)
        proxy_student_logits = self.proxy_student(s_feat.detach())

        # Image part
        t_feat, teacher_logits = self.get_image_features(image_data)
        proxy_teacher_logits = self.proxy_teacher(t_feat.detach())

        return {
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "proxy_student_logits": proxy_student_logits,
            "proxy_teacher_logits": proxy_teacher_logits,
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
    model = ResnetC2KDModel(
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
