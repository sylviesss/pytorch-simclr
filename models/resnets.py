from torchvision import models
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module


class ResnetEncoder(models.resnet.ResNet):
    def __init__(self,
                 block=models.resnet.Bottleneck,
                 layers=[3, 4, 6, 3],
                 low_quality_img=True):
        super(ResnetEncoder, self).__init__(block=block, layers=layers)
        self.low_quality_img = low_quality_img

        if self.low_quality_img:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.bn1 = self._norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.low_quality_img:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x


# For removing layers
class Identity(nn.Module):
    def __init__(self, need_to_flatten=False):
        super(Identity, self).__init__()
        self.flatten_flag = need_to_flatten
        self.flatten = nn.Flatten()

    def forward(self, x):
        if self.flatten_flag:
            x = self.flatten(x)
        else:
            pass
        return x


class ResnetSupervised(models.resnet.ResNet):
    def __init__(self,
                 blocks=models.resnet.Bottleneck,
                 layers=[3, 4, 6, 3],
                 low_quality_img=True,
                 n_class=10):
        super(ResnetSupervised, self).__init__(block=blocks,
                                               layers=layers,
                                               num_classes=n_class)
        self.low_quality_img = low_quality_img
        if self.low_quality_img:
            self.conv1 = nn.Conv2d(3, 64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
            self.bn1 = self._norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.low_quality_img:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Modifying the pytorch dropout module
class DropoutNd(Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class Dropout(DropoutNd):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always keep training = True
        return F.dropout(x, self.p, True, self.inplace)


class ResnetEncoderDropout(models.resnet.ResNet):
    def __init__(self,
                 drop_prob,
                 block=models.resnet.Bottleneck,
                 layers=[3, 4, 6, 3],
                 low_quality_img=True):
        super(ResnetEncoderDropout, self).__init__(block=block, layers=layers)
        self.low_quality_img = low_quality_img
        self.p = drop_prob

        if self.low_quality_img:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.bn1 = self._norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = Dropout(p=self.p, inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        if not self.low_quality_img:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)

        x = self.avgpool(x)

        return x