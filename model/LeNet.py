# ======================
# file: LeNet.py
# author: KONI-code2023@github
# date: 2024-01-16
# ======================
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        
        # 第一個卷積層，輸入通道數為1，輸出通道數為6，卷積核大小為5x5，填充為2
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 第二個卷積層，輸入通道數為6，輸出通道數為16，卷積核大小為5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 定義全連接層
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu7 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # 通過第一個卷積層
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.avgpool1(x)

        # 通過第二個卷積層
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool2(x)
        
        # 展平
        x = self.flatten(x)
        
        # 通過全連接層
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.fc3(x)
        
        return x
