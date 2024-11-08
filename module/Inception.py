import torch
import torch.nn as nn
import torch.nn.functional as F

# Inception 모듈 정의
class Inception(nn.Module): # in_channels는 Inception 모듈의 입력 채널수를 의미
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_redue, ch5x5, pool_proj):
        super(Inception).__init__()
        
        # 1x1 conv layer
        self.layer1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        
        # 1x1 -> 3x3 conv layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
        )
        
        # 1x1 -> 5x5 conv layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_redue, kernel_size=1),
            nn.Conv2d(ch5x5_redue, ch5x5, kernel_size=5, padding=2)
        )
        
        # 3x3 maxpooling -> 1x1 layer
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
        
    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(x)
        layer4 = self.layer4(x)
        
        # outputs을 지정해 4개의 레이어를 합쳐서 결과 반환
        outputs = [layer1, layer2, layer3, layer4]
        return torch.cat(outputs, 1)
        # torch.cat을 사용해 4개의 레이어에서 나온값을 1차원으로 연결