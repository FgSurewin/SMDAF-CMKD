import torch
import torch.nn as nn

from models.audio_base.ast_model import ASTModel
from models.image_base.image_model import ImageModel

from models.lightning_module.audio_image_kd_base_model import (
    ImageAudioKDCompositeBaseModel,
)

from models.basic_modules.proxy_module import Proxy


class ImageAudioASTKDModel(ImageAudioKDCompositeBaseModel):
    def __init__(
        self,
        num_classes,
        label_names,
        kd_config,
        fstride=10,
        tstride=10,
        input_fdim=64,
        input_tdim=173,
        imagenet_pretrain=True,
        audioset_pretrain=False,
        model_size="tiny224",
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes, label_names=label_names, kd_config=kd_config
        )
        self.ast_model = ASTModel(
            label_dim=num_classes,
            label_names=label_names,
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            model_size=model_size,
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
        for param in self.image_model.parameters():
            param.requires_grad = False

        # Extract the pre-trained weights and bias for the fully connected layer
        pretrained_fc_weights = self.image_model.img_extractor.fc.weight.data.clone()
        pretrained_fc_bias = self.image_model.img_extractor.fc.bias.data.clone()
        dummy_teacher_feat, _ = self.get_image_features(torch.rand((5, 3, 360, 480)))
        self.proxy_teacher = Proxy(
            dummy_teacher_feat, num_classes, pretrained_fc_weights, pretrained_fc_bias
        )
        # dummy_student_feat, _ = self.get_audio_features(torch.rand((5, 2, 64, 173)))
        # self.proxy_student = Proxy(dummy_student_feat, num_classes)
        self.save_hyperparameters()

    def get_image_features(self, image_data):
        # test_data = torch.rand((5, 3, 360, 480))  # dummy data
        x = self.image_model.img_extractor.conv1(image_data)
        x = self.image_model.img_extractor.bn1(x)
        x = self.image_model.img_extractor.relu(x)
        f0 = x

        x = self.image_model.img_extractor.layer1(x)  # 32x32
        f1 = x
        x = self.image_model.img_extractor.layer2(x)  # 16x16
        f2 = x
        x = self.image_model.img_extractor.layer3(x)  # 8x8
        f3 = x

        x = self.image_model.img_extractor.layer4(x)
        f4 = x

        x = self.image_model.img_extractor.avgpool(x)
        x = x.view(x.size(0), -1)
        f5 = x
        x = self.image_model.img_extractor.fc(x)
        feat_t = [f0, f1, f2, f3, f4, f5]
        feat_t = [f.detach() for f in feat_t]
        return feat_t, x

    def parse_batch(self, batch):
        audio_data, image_data, labels = batch
        return {"audio_data": audio_data, "image_data": image_data, "labels": labels}

    def forward(self, data):
        audio_data = data["audio_data"]
        image_data = data["image_data"]
        labels = data["labels"]
        # Audio part
        student_logits = self.ast_model(audio_data)
        # student_logits = self.audio_model(audio_data)
        # stu_feat, _ = self.get_audio_features(audio_data)
        # proxy_student_logits = self.proxy_student(stu_feat)

        # Image part
        feat, teacher_logits = self.get_image_features(image_data)
        proxy_teacher_logits = self.proxy_teacher(feat)

        return {
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            # "proxy_student_logits": proxy_student_logits,
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
    model = ImageAudioASTKDModel(
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
