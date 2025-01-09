import torch
import torch.nn as nn
from models.lightning_module.single_base_model import SingleBaseModel
from models.basic_modules.extractor_module import ViTFeatureExtractor
import torchvision.models as models


class ImageModel(SingleBaseModel):
    """
    The Image model.
    :param extractor_type: you can choose either "resnet" or "vit"
    :param model_size: it is used to create pre-trained vit model
    """

    def __init__(
        self,
        num_classes: int = 7,
        label_names=[],
        imagenet_pretrain: bool = True,
        extractor_type: str = "resnet",  # resnet, vit
        model_size: str = "resnet50",  # resnet18, base224
        verbose=True,
        **kwargs,
    ):
        super().__init__(num_classes, label_names)
        self.extractor_type = extractor_type
        if extractor_type == "resnet":
            # self.process_data = nn.Sequential(nn.BatchNorm2d(3), nn.ReLU())
            self.img_extractor = self.get_resnet(model_size)
            self.img_extractor.fc = nn.Linear(
                self.img_extractor.fc.in_features, num_classes
            )
        elif extractor_type == "vit":
            # self.process_data = nn.Identity()
            self.img_extractor = ViTFeatureExtractor(
                imagenet_pretrain=imagenet_pretrain,
                model_size=model_size,
                verbose=verbose,
            )
            self.classify_fc = nn.Sequential(
                nn.LayerNorm(self.img_extractor.original_embedding_dim),
                nn.Linear(self.img_extractor.original_embedding_dim, num_classes),
            )

        self.kwargs = kwargs
        self.save_hyperparameters()

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

    def forward(self, x):
        if self.extractor_type == "resnet":
            logits = self.img_extractor(x)
        elif self.extractor_type == "vit":
            x_cls, x_dist = self.img_extractor(x)
            features = (x_cls + x_dist) / 2
            logits = self.classify_fc(features)
        return logits


if __name__ == "__main__":
    # Set global parameters
    NUM_CLASSES = 7
    # Create model
    model = ImageModel(num_classes=NUM_CLASSES, extractor_type="vit")
    # Create dummy input
    # (batch_size, channels, n_mels, seq_len)
    # x = torch.rand((10, 3, 360, 480))
    x = torch.rand((10, 3, 224, 224))
    # Forward pass
    y = model(x)
    print(y.shape)
