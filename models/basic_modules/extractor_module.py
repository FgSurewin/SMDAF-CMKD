import torch
import timm
import torch.nn as nn
import torchvision.models as models


class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        freeze_layer_num=0,
        output_features=512,
        fc_num=128,
        dropout=0.5,
        imagenet_pretrain=True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.extractor = self.get_extractor(freeze_layer_num, imagenet_pretrain)
        self.extractor.fc = nn.Linear(
            in_features=self.extractor.fc.in_features, out_features=output_features
        )
        self.bn_extractor_fc = nn.BatchNorm1d(output_features)
        self.fc = nn.Linear(in_features=output_features, out_features=fc_num)
        self.bn_fc = nn.BatchNorm1d(fc_num)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.kwargs = kwargs

    def forward(self, x):
        # Original data -> output(512)
        out = self.extractor(x)
        out = self.relu(self.bn_extractor_fc(out))
        out = self.dropout(out)

        # output(512) -> fc_num(128)
        out = self.fc(out)
        out = self.relu(self.bn_fc(out))
        out = self.dropout(out)
        return out

    def get_extractor(self, freeze_layer_num=0, imagenet_pretrain=True):
        if imagenet_pretrain:
            extracter = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            extracter = models.resnet50()
        if freeze_layer_num > 0:
            for param in extracter.layer1.parameters():
                param.requires_grad = False
        if freeze_layer_num > 1:
            for param in extracter.layer2.parameters():
                param.requires_grad = False
        if freeze_layer_num > 2:
            for param in extracter.layer3.parameters():
                param.requires_grad = False
        if freeze_layer_num > 3:
            for param in extracter.layer4.parameters():
                param.requires_grad = False
        return extracter


class ViTFeatureExtractor(nn.Module):
    def __init__(
        self, imagenet_pretrain=True, model_size="base224", verbose=True, **kwargs
    ):
        super().__init__()
        assert (
            timm.__version__ == "0.4.5"
        ), "Please use timm == 0.4.5, the code might not be compatible with newer versions."

        if verbose == True:
            print("---------------ViT Extractor Summary---------------")
            print(f"ImageNet pretraining: {imagenet_pretrain}")
            print(f"The pre-trained ViT model size is {model_size}")

        self.v = self.get_extractor(model_size, imagenet_pretrain)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.kwargs = kwargs

    def get_extractor(self, model_size, imagenet_pretrain):
        if model_size == "tiny224":
            v = timm.create_model(
                "vit_deit_tiny_distilled_patch16_224", pretrained=imagenet_pretrain
            )
        elif model_size == "small224":
            v = timm.create_model(
                "vit_deit_small_distilled_patch16_224", pretrained=imagenet_pretrain
            )
        elif model_size == "base224":
            v = timm.create_model(
                "vit_deit_base_distilled_patch16_224", pretrained=imagenet_pretrain
            )
        elif model_size == "base384":
            v = timm.create_model(
                "vit_deit_base_distilled_patch16_384", pretrained=imagenet_pretrain
            )
        else:
            raise Exception(
                "Model size must be one of tiny224, small224, base224, base384."
            )

        return v

    def forward(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x_cls = x[:, 0]
        x_dist = x[:, 1]
        return x_cls, x_dist
