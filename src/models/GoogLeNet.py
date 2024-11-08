############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.Inception import Inception # Inception 모듈가져오기
############################################################

# Auxiliary Classifier 
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        
        # 5x5 pooling을 적용해 크기를 줄임
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        
        # 1x1 conv으로 채널 수를 줄임
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        
        #FC layer 
        self.fc1 = nn.Linear(128 * 4 * 4, 1024) # 특성 크기에 맞춰 크기를 조정
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1) # 평탄화
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)
        return x
      
