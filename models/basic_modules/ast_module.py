import timm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from timm.models.layers import to_2tuple, trunc_normal_


"""
AST: Audio Spectrogram Transformer
Code is from: https://github.com/YuanGongND/ast
"""


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModule(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """

    def __init__(
        self,
        # label_dim=7,
        extra_dim=192,
        has_bootster_patch=False,
        fstride=10,
        tstride=10,
        input_fdim=64,
        input_tdim=173,  # 1 second
        imagenet_pretrain=True,
        audioset_pretrain=False,
        model_size="base224",
        verbose=True,
    ):

        super(ASTModule, self).__init__()
        assert (
            timm.__version__ == "0.4.5"
        ), "Please use timm == 0.4.5, the code might not be compatible with newer versions."

        if verbose == True:
            print("---------------AST Model Summary---------------")
            print(
                "ImageNet pretraining: {:s}, AudioSet pretraining: {:s}".format(
                    str(imagenet_pretrain), str(audioset_pretrain)
                )
            )
            print(f"The pre-trained ViT model size is {model_size}")
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # Determine if the AST include booster or not
        self.has_bootster_patch = has_bootster_patch

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            self.v = self.get_vit_model(model_size, imagenet_pretrain)
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches**0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            # self.mlp_head = nn.Sequential(
            #     nn.LayerNorm(self.original_embedding_dim),
            #     nn.Linear(self.original_embedding_dim, label_dim),
            # )

            if self.has_bootster_patch:
                self.extra_proj = nn.Linear(
                    extra_dim, self.original_embedding_dim
                )  # CHECK: Apply linear transfomation to engineering data
            else:
                self.extra_proj = None

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                # print("original_num_patches: ", self.original_num_patches)
                # print("oringal_hw: ", self.oringal_hw)
                # print("original_embedding_dim: ", self.original_embedding_dim)
                print(
                    "frequncey stride={:d}, time stride={:d}".format(fstride, tstride)
                )
                print("number of patches={:d}".format(num_patches))
                print("old proj.weight: ", self.v.patch_embed.proj.weight.shape)
            # the linear projection layer
            new_proj = torch.nn.Conv2d(
                1,
                self.original_embedding_dim,
                kernel_size=(16, 16),
                stride=(fstride, tstride),
            )
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
                )
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if verbose == True:
                print("new proj.weight: ", new_proj.weight.shape)

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = (
                    self.v.pos_embed[:, 2:, :]
                    .detach()
                    .reshape(1, self.original_num_patches, self.original_embedding_dim)
                    .transpose(1, 2)
                    .reshape(
                        1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw
                    )
                )
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        :,
                        int(self.oringal_hw / 2)
                        - int(t_dim / 2) : int(self.oringal_hw / 2)
                        - int(t_dim / 2)
                        + t_dim,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(self.oringal_hw, t_dim), mode="bilinear"
                    )
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        int(self.oringal_hw / 2)
                        - int(f_dim / 2) : int(self.oringal_hw / 2)
                        - int(f_dim / 2)
                        + f_dim,
                        :,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                    )
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(
                    1, self.original_embedding_dim, num_patches
                ).transpose(1, 2)

                if self.has_bootster_patch:
                    # CHECK: add new positional embedding for the extra patch
                    extra_pos_embed = nn.Parameter(
                        torch.zeros(1, 1, self.original_embedding_dim)
                    )
                    trunc_normal_(extra_pos_embed, std=0.02)
                    # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                    self.v.pos_embed = nn.Parameter(
                        torch.cat(
                            [
                                self.v.pos_embed[:, :2, :].detach(),
                                new_pos_embed,
                                extra_pos_embed,
                            ],
                            dim=1,
                        )
                    )
                else:
                    # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                    self.v.pos_embed = nn.Parameter(
                        torch.cat(
                            [
                                self.v.pos_embed[:, :2, :].detach(),
                                new_pos_embed,
                            ],
                            dim=1,
                        )
                    )

            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                if has_bootster_patch:
                    number_patches = self.v.patch_embed.num_patches + 3
                else:
                    number_patches = self.v.patch_embed.num_patches + 2
                new_pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        number_patches,
                        self.original_embedding_dim,
                    )
                )  # CHECK: +3 instead of +2
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=0.02)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def get_vit_model(self, model_size, imagenet_pretrain):
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

    @autocast()
    def forward(self, x, eng_data=None):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        # x = x.unsqueeze(1)
        # x = x.transpose(2, 3)
        # CHECK: Our input x = (B, C, n_mel, time_frame_num)
        x = x.mean(dim=1, keepdim=True)  # x = (B, 1, 64, 173)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        # CHECK: expect eng_data = (B, 52)
        if self.has_bootster_patch and eng_data is not None:
            eng_data_patch = self.extra_proj(eng_data).unsqueeze(
                1
            )  # expect eng_data_patch = (B, 1, 768)
            # CHECK: concat new patch
            x = torch.cat((cls_tokens, dist_token, x, eng_data_patch), dim=1)
        else:
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x_cls = x[:, 0]
        x_dist = x[:, 1]
        # x = (x[:, 0] + x[:, 1]) / 2

        # x = self.mlp_head(x)
        return x_cls, x_dist


if __name__ == "__main__":
    # Set global parameters
    # NUM_CLASSES = 10
    # Create model
    model = ASTModule(extra_dim=192, has_bootster_patch=True)
    # Create dummy input
    # (batch_size, channels, n_mels, seq_len)
    eng_data = torch.rand([10, 192])
    audio_data = torch.randn(10, 2, 64, 173)
    # Forward pass
    x_cls, x_dist = model(audio_data, eng_data)
    # y = model(audio_data, gaf_data)
    print(x_cls.shape, x_dist.shape)
