############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module.Inception import Inception # Inception 모듈가져오기
############################################################

# Auxiliary Classifier 
# 네트워크의 중간중간에 Classifier를 추가하여 학습 Loss를 흐르게 만들어주는 방법입니다. 
# 마치 긴 터널의 중간중간에 환기구를 뚫어 공기 흐름을 원활하게 해 주는 것과 같은 이치라고 할 수 있습니다.
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
      
# GoogLeNet 모델 정의
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        
        
        #초기 합성곱 계층
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride= 2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Inception 모듈 추가
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        
        # 보조 분류기 추가 (훈련 시만 사용)
        self.aux1 = AuxiliaryClassifier(512, num_classes)
        self.aux2 = AuxiliaryClassifier(528, num_classes)

        # 최종 풀링과 완전 연결 계층
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # 초기 conv layer
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        
        # Inception module
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        
        # 첫 번째 보조 분류기 (훈련 시만 사용)
        aux1 = self.aux1(x) if self.training else None
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        # 두 번째 보조 분류기 (훈련 시만 사용)
        aux2 = self.aux2(x) if self.training else None
        
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # 최종 풀링과 완전 연결 계층
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        # 학습 시, 메인 출력과 보조 분류기 출력을 반환
        return x, aux1, aux2 if self.training else x