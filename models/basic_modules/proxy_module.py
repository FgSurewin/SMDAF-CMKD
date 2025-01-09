import torch
import torch.nn as nn
import torch.nn.functional as F


def depthwise_separable_conv(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_planes,
            bias=False,
        ),
        nn.BatchNorm2d(in_planes),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        ),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

"""
Shadow Knowledge Distillation: Bridging Offline and Online Knowledge Transfer
Code is from: https://github.com/lliai/SHAKE
"""



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        # nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def conv_1x1_bn(num_input_channels, num_mid_channel):
    return nn.Sequential(
        conv1x1(num_input_channels, num_mid_channel),
        nn.BatchNorm2d(num_mid_channel),
        nn.ReLU(inplace=True),
    )


class Proxy(nn.Module):
    def __init__(self, feat_t):
        super(Proxy, self).__init__()

        # self.fuse1 = depthwise_separable_conv(feat_t[1].shape[1], feat_t[2].shape[1], 2)
        self.fuse1 = conv_bn(feat_t[1].shape[1], feat_t[2].shape[1], 2)
        self.fuse2 = conv_1x1_bn(feat_t[2].shape[1], feat_t[2].shape[1])
        # self.fuse3 = depthwise_separable_conv(feat_t[2].shape[1], feat_t[3].shape[1], 2)
        self.fuse3 = conv_bn(feat_t[2].shape[1], feat_t[3].shape[1], 2)
        self.fuse4 = conv_1x1_bn(feat_t[3].shape[1], feat_t[3].shape[1])
        # self.fuse5 = depthwise_separable_conv(feat_t[3].shape[1], feat_t[4].shape[1], 2)
        # self.fuse5 = conv_bn(feat_t[3].shape[1], feat_t[3].shape[1], 1)
        self.fuse5 = conv_bn(feat_t[3].shape[1], feat_t[4].shape[1], 2)
        # self.fuse6 = conv_1x1_bn(feat_t[4].shape[1], feat_t[4].shape[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # self.fc = nn.Linear(feat_t[4].shape[1], num_classes)
        # # Initialize the fc layer with pre-trained weights and bias
        # if pretrained_fc_bias is not None and pretrained_fc_weights is not None:
        #     self.fc.weight.data.copy_(pretrained_fc_weights)
        #     self.fc.bias.data.copy_(pretrained_fc_bias)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000000

    def forward(self, f, weight, bias):
        # x = self.fuse5(
        #     self.fuse3(self.fuse1(f[1]) + self.fuse2(f[2])) + self.fuse4(f[3])
        # )
        x = self.fuse5(
            self.fuse3(self.fuse1(f[0]) + self.fuse2(f[1])) + self.fuse4(f[2])
        )

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.linear(x, weight, bias)
        # x = self.fc(x)
        return x


class C2KDProxy(nn.Module):
    def __init__(self, feat_t, num_classes):
        super(C2KDProxy, self).__init__()
        self.adaption_layer = conv_1x1_bn(feat_t.shape[1], feat_t.shape[1])
        self.fc = nn.Linear(feat_t.shape[1], num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.fuse5(
        #     self.fuse3(self.fuse1(f[1]) + self.fuse2(f[2])) + self.fuse4(f[3])
        # )
        x = self.adaption_layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
