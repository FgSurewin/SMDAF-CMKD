import torch
import torch.nn as nn
from models.lightning_module.single_base_model import SingleBaseModel
import torchvision.models as models


class ResNetAudioModel(SingleBaseModel):
    def __init__(
        self,
        num_classes: int = 7,
        label_names=[],
        resnet_type="resnet18",
        is_mean_upsample=False,
        **kwargs,
    ):
        super().__init__(num_classes, label_names)
        self.is_mean_upsample = is_mean_upsample
        self.upsample = nn.Conv2d(
            in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1
        )
        self.extractor = self.get_resnet(resnet_type)
        self.extractor.fc = nn.Linear(self.extractor.fc.in_features, num_classes)
        self.kwargs = kwargs
        self.save_hyperparameters()

    def forward(self, x):
        out = self.mean_upsample(x) if self.is_mean_upsample else self.upsample(x)
        return self.extractor(out)

    def get_resnet(self, resnet_type):
        if resnet_type == "resnet18":
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif resnet_type == "resnet34":
            return models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif resnet_type == "resnet50":
            return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_type == "resnet50":
            return models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            raise ValueError(f"The resnet_type {resnet_type} is not valid!")

    def mean_upsample(self, x):
        mean_channel = torch.mean(x, dim=1, keepdim=True)
        final = torch.cat([x, mean_channel], dim=1)
        return final


if __name__ == "__main__":
    # Set global parameters
    NUM_CLASSES = 7
    # Create model
    model = ResNetAudioModel(num_classes=NUM_CLASSES)
    # Create dummy input
    # (batch_size, channels, n_mels, seq_len)
    x = torch.rand((10, 2, 64, 173))
    # x = torch.rand((10, 3, 224, 224))
    # Forward pass
    y = model(x)
    print(y.shape)
