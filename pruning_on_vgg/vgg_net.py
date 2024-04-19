import math
from typing import Dict, List, Union

import torch.nn as nn

cfgs: Dict[int, List[Union[str, int]]] = {
    11: [64,     "M", 128,      "M", 256, 256,           "M", 512, 512,           "M", 512, 512],
    13: [64, 64, "M", 128, 128, "M", 256, 256,           "M", 512, 512,           "M", 512, 512],
    16: [64, 64, "M", 128, 128, "M", 256, 256, 256,      "M", 512, 512, 512,      "M", 512, 512, 512],
    19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=11, cfg=None, init_weights=True):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = cfgs[depth]
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, l, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(l), nn.ReLU(inplace=True)]
                in_channels = l
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()




