import torch.nn as nn
import torch.nn.functional as F

class BFP(nn.Module):

    def __init__(self,
                in_channels,
                num_levels,
                refine_level=2,
                conv_cfg=None,
                norm_cfg=None):
        super(BFP, self).__init__()
        # assert refine_type in [None, 'conv', 'non_local']
        
        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        assert 0 <= self.refine_level < self.num_levels

        self.refine = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1)

    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, inputs):

        features = []
        gather_size = inputs[self.refine_level].size()[2:]

        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(inputs[i], size=gather_size, mode='nearest')
            features.append(gathered)

        bsf = sum(features) / len(features)

        bsf = self.refine(bsf)

        outputs = []

        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]

            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)

            outputs.append(residual + inputs[i])

        return outputs