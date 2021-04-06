from torchvision import models
from torch import nn


class ResnetEncoder(models.resnet.ResNet):
    def __init__(self,
                 block=models.resnet.Bottleneck,
                 layers=[3, 4, 6, 3],
                 cifar=True):
        super(ResnetEncoder, self).__init__(block=block, layers=layers)
        self.cifar = cifar

        if self.cifar:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.bn1 = self._norm_layer(64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x