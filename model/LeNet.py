# ======================
# file: LeNet.py
# author: KONI-code2023@github
# comments: ChatGPT
# date: 2024-01-16
# ======================
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        
        # 第一個卷積層，輸入通道數為1，輸出通道數為6，卷積核大小為5x5，填充為2
        self.conv1 = self.conv_layer(1, 6, kernel_size=5, padding=2)
        
        # 第二個卷積層，輸入通道數為6，輸出通道數為16，卷積核大小為5x5
        self.conv2 = self.conv_layer(6, 16, kernel_size=5)

        # 定義全連接層
        self.fc_layers(num_classes)

    def conv_layer(self, in_channels, out_channels, kernel_size, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def fc_layers(self, num_classes):
        # 展平操作，將多維的輸入拉平成一維
        self.flatten = nn.Flatten()
        
        # 第一個全連接層，輸入特徵數為16x5x5，輸出特徵數為120
        self.fc1 = self.fc_layer(16 * 5 * 5, 120)
        
        # 第二個全連接層，輸入特徵數為120，輸出特徵數為84
        self.fc2 = self.fc_layer(120, 84)
        
        # 輸出層，輸入特徵數為84，輸出特徵數為num_classes
        self.fc3 = nn.Linear(84, num_classes)

    def fc_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通過第一個卷積層
        x = self.conv1(x)
        
        # 通過第二個卷積層
        x = self.conv2(x)
        
        # 展平
        x = self.flatten(x)
        
        # 通過全連接層
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
