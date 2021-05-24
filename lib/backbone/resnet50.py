import torch
from torch import nn
import torch.nn.functional as F
import sys
from typing import List

sys.path.insert(0, 'lib')
from layers.batch_norm import FrozenBatchNorm2d

class Bottleneck(nn.Module):
    def __init__(self, 
                in_cha : int, 
                neck_cha : int, 
                out_cha : int, 
                stride : int, 
                has_bias : bool =False):
        super(Bottleneck, self).__init__()

        self.downsample = None
        if in_cha!= out_cha or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, kernel_size=1, stride=stride, bias=has_bias),
                FrozenBatchNorm2d(out_cha),)

        self.conv1 = nn.Conv2d(in_cha, neck_cha, kernel_size=1, stride=1, bias=has_bias)
        
        # FrozenBatchNorm2d : batch size가 매우 작은 경우, batch statistics가 큰 영향을 끼치지 못하고 오히려 성능을 감소시킨다. 
        # 또한 다수의 GPU를 사용할 때, 각각의 GPU에서 batch statistics가 연산되지 않아, 오직 하나의 GPU에서만 batch norm을 수행한다고 한다.
        self.bn1 = FrozenBatchNorm2d(neck_cha)

        self.conv2 = nn.Conv2d(neck_cha, neck_cha, kernel_size=3, stride=stride, padding=1, bias=has_bias)
        self.bn2 = FrozenBatchNorm2d(neck_cha)

        self.conv3 = nn.Conv2d(neck_cha, out_cha, kernel_size=1, bias=has_bias)
        self.bn3 = FrozenBatchNorm2d(out_cha)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu_(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = F.relu_(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, 
                 freeze_at, 
                 has_bias = False):
        super(ResNet50, self).__init__()

        self.has_bias = has_bias
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=has_bias)
        self.bn1 = FrozenBatchNorm2d(64)

        block_counts = [3, 4, 6, 3]
        bottleneck_channels_list = [64, 128, 256, 512]
        out_channels_list = [256, 512, 1024, 2048]
        stride_list = [1, 2, 2, 2]
        in_channels = 64

        self.layer1 = self._make_layer(block_counts[0], 64, bottleneck_channels_list[0], out_channels_list[0], stride_list[0])
        self.layer2 = self._make_layer(block_counts[1], out_channels_list[0], bottleneck_channels_list[1], out_channels_list[1], stride_list[1])
        self.layer3 = self._make_layer(block_counts[2], out_channels_list[1], bottleneck_channels_list[2], out_channels_list[2], stride_list[2])
        self.layer4 = self._make_layer(block_counts[3], out_channels_list[2], bottleneck_channels_list[3], out_channels_list[3], stride_list[3])

        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.kaiming_normal_(l.weight, mode='fan_out')
                if self.has_bias:
                    nn.init.constant_(l.bias, 0)

        self._freeze_backbone(freeze_at)

    def _make_layer(self, 
                    num_blocks : int, 
                    in_channels : int, 
                    bottleneck_channels : int, 
                    out_channels : int, 
                    stride : int):

        layers = []
        for _ in range(num_blocks):
            layers.append(Bottleneck(in_channels, bottleneck_channels, out_channels, stride, self.has_bias))
            stride = 1
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _freeze_backbone(self, 
                        freeze_at : int):

        if freeze_at < 0:
            return
        if freeze_at >= 1:
            for p in self.conv1.parameters():
                p.requires_grad = False
        if freeze_at >= 2:
            for p in self.layer1.parameters():
                p.requires_grad = False
        if freeze_at >= 3:
            print("Freeze too much layers! Only freeze the first 2 layers.")

    def forward(self, x):

        outputs = []
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # blocks
        x = self.layer1(x)
        outputs.append(x)
        x = self.layer2(x)
        outputs.append(x)
        x = self.layer3(x)
        outputs.append(x)
        x = self.layer4(x)
        outputs.append(x)

        return outputs

if __name__ == "__main__":
    # input : [batch size, channel, height, width]
    input = torch.randn((2, 3, 64, 64))
    model = ResNet50(2, False)
    output = model(input)

    """
    Output : 서로 다른 크기의 feature map을 요소로 가지는 list
    
    torch.Size([2, 256, 16, 16])
    torch.Size([2, 512, 8, 8])
    torch.Size([2, 1024, 4, 4])
    torch.Size([2, 2048, 2, 2])
    """
    for ot in output:
        print(ot.shape)