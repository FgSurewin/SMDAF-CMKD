import torch
import torch.nn as nn
from models.lightning_module.single_base_model import SingleBaseModel
from models.basic_modules.ast_module import ASTModule


class ASTModel(SingleBaseModel):
    def __init__(
        self,
        num_classes: int = 7,
        label_names: list = [],
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 64,
        input_tdim: int = 173,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = False,
        model_size: str = "base384",
        **kwargs,
    ):
        super().__init__(num_classes, label_names)
        self.ast_module = ASTModule(
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            model_size=model_size,
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.ast_module.original_embedding_dim),
            nn.Linear(self.ast_module.original_embedding_dim, num_classes),
        )
        self.kwargs = kwargs
        self.save_hyperparameters()

    def forward(self, x):
        x_cls, x_dist = self.ast_module(x)
        features = (x_cls + x_dist) / 2

        logits = self.mlp_head(features)
        return logits


if __name__ == "__main__":
    NUM_CLASSES = 7
    tdim = 173
    fdim = 64
    model = ASTModel(
        num_classes=NUM_CLASSES,
        input_fdim=fdim,
        input_tdim=tdim,
    )
    x = torch.rand([10, 2, fdim, tdim])
    logits = model(x)
    print(logits.shape)
