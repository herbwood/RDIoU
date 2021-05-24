import math
import sys
from typing import List

sys.path.insert(0, 'lib')

import torch
from torch import nn
import torch.nn.functional as F
from .resnet50 import ResNet50

class FPN(nn.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """
    def __init__(self, 
                bottom_up, # backbone network 
                layers_begin, 
                layers_end):
        super(FPN, self).__init__()

        assert layers_begin > 1 and layers_begin < 6
        assert layers_end > 4 and layers_begin < 8

        # FPN이 입력받는 feature map의 channel 
        # output feature map의 channel 
        # channel 크기가 작은 순서대로 입력받음 
        in_channels = [256, 512, 1024, 2048]
        fpn_dim = 256
        in_channels = in_channels[layers_begin-2:]

        # Backbone에서 얻은 feature map에 적용할 conv layer를 저장할 ModuleList 
        # lateral conv layer -> output conv layer로 channel 수가 fpn_dim인 feature map 출력되도록 함 
        lateral_convs = nn.ModuleList()
        output_convs = nn.ModuleList()

        for idx, in_channels in enumerate(in_channels):
            lateral_conv = nn.Conv2d(in_channels, fpn_dim, kernel_size=1)
            output_conv = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)

            # normalization
            nn.init.kaiming_normal_(lateral_conv.weight, mode='fan_out')
            nn.init.constant_(lateral_conv.bias, 0)
            nn.init.kaiming_normal_(output_conv.weight, mode='fan_out')
            nn.init.constant_(output_conv.bias, 0)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # bottom-up이기 때문에 lateral conv layer, output conv가 
        # 해상도가 큰 feature map부터 적용될 수 있도록 하기 위해 역순으로 바꿔줌 
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.bottom_up = bottom_up
        self.output_b = layers_begin
        self.output_e = layers_end

        if self.output_e == 7:
            self.p6 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)
            self.p7 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)

            for l in [self.p6, self.p7]:
                nn.init.kaiming_uniform_(l.weight, a=1)  # pyre-ignore
                nn.init.constant_(l.bias, 0)

    def forward(self, x):
        
        # tensor를 backbone network에 입력하여 얻은 List[Tensor]
        """
        Output 예시)
        bottom_up_features : [
            torch.Size([2, 256, 16, 16])
            torch.Size([2, 512, 8, 8])
            torch.Size([2, 1024, 4, 4])
            torch.Size([2, 2048, 2, 2])
        ]
        """
        # FPN을 설계할  layer부터 슬라이싱함
        # 역순으로 바꿔줌 
        bottom_up_features = self.bottom_up(x)
        bottom_up_features = bottom_up_features[self.output_b - 2:]
        bottom_up_features = bottom_up_features[::-1]

        results = []

        # 가장 작은 크기의 feature map에 대하여 (upsampling 수행 없이) lateral conv에 입력
        # 이를 통해 얻은 feature map을 results에 저장 
        prev_features = self.lateral_convs[0](bottom_up_features[0])
        results.append(self.output_convs[0](prev_features))

        # 1) 상위 층의 feature map을 2배로 upsampling
        # 2) backbone network를 통과하여 얻은 feature map
        # 1), 2)를 element-wise하게 더해줌 
        # results 리스트에 저장 
        for l_id, (features, lateral_conv, output_conv) in enumerate(zip(bottom_up_features[1:], self.lateral_convs[1:], self.output_convs[1:])):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest", align_corners=False)
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            results.append(output_conv(prev_features))

        # end layer에 따라 추가적인 max pooling 혹은 conv layer 적용 
        if(self.output_e == 6):
            p6 = F.max_pool2d(results[0], kernel_size=1, stride=2, padding=0)
            results.insert(0, p6)

        elif(self.output_e == 7):
            p6 = self.p6(results[0])
            results.insert(0, p6)
            p7 = self.p7(F.relu(results[0]))
            results.insert(0, p7)

        # channel 수가 모두 fpn_dim(=256)인 feature map으로 구성된 리스트 
        """
        results 예시)
        
        [torch.Size([2, 256, 16, 16])
        torch.Size([2, 256, 8, 8])
        torch.Size([2, 256, 4, 4])
        torch.Size([2, 256, 2, 2])]
        """
        return results