import torch.nn as nn
import torch.nn.functional as F

class BFP(nn.Module):

    def __init__(self,
                in_channels,
                num_levels,
                refine_level=2):
        super(BFP, self).__init__()
        
        self.in_channels = in_channels
        self.num_levels = num_levels

        self.refine_level = refine_level
        assert 0 <= self.refine_level < self.num_levels

        # refine시켜주는 conv layer 
        self.refine = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1)

    # normalization 
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, inputs):
        
        # 입력으로 feature map으로 구성된 리스트 
        # 그 중 중간 크기의 해상도를 가지는 feature map이 기준이 됨 
        # refine level 인자를 통해 결정 
        # gather_size는 기준 feature map의 height, width 
        features = []
        gather_size = inputs[self.refine_level].size()[2:]

        # 기준 feature map보다 크면 max pool
        # 작으면 interpolate하여 
        # input 리스트에 저장된 feature map이 모두 같은 해상도가 되도록 함 
        # features 리스트에는 같은 해상도와 같은 channel을 가지는 feature map이 저장됨 
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(inputs[i], size=gather_size, mode='nearest')
            features.append(gathered)

        # features에 저장된 feature map을 모두 element-wise하게 더한 후 개수만큼 나눠줌 
        # refine시켜주는 conv layer를 적용함
        # 이를 통해 balanced feature map을 얻음 
        bsf = sum(features) / len(features)
        bsf = self.refine(bsf)

        outputs = []

        # balanced feature map을 각각 원래 크기로 돌려준 후 
        # element-wise하게 더해줌 
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]

            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)

            outputs.append(residual + inputs[i])

        # Balanced feature map List[Tensor]
        return outputs