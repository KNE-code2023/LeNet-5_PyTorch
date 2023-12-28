import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 定義卷積層
        self.conv1 = self.conv_layer(1, 6, kernel_size=5, padding=2)
        self.conv2 = self.conv_layer(6, 16, kernel_size=5)

        # 定義全連接層
        self.fc_layers(num_classes)

    def conv_layer(self, in_channels, out_channels, kernel_size, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def fc_layers(self, num_classes):
        self.flatten = nn.Flatten()
        self.fc1 = self.fc_layer(16 * 5 * 5, 120)
        self.fc2 = self.fc_layer(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def fc_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            #nn.Sigmoid()
            nn.ReLU()
        )

    def forward(self, x):
        # 卷積層
        x = self.conv1(x)
        x = self.conv2(x)

        # 攤平
        x = self.flatten(x)

        # 全連接層
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
