import torch.nn as nn
import torch.nn.functional as F

class SpeechRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 按原来命名
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # 固定输出 8×8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  # conv1 → relu → pool1
        x = self.pool2(self.relu2(self.conv2(x)))  # conv2 → relu → pool2
        x = self.adaptive_pool(x)                 #  → [B,64,8,8]
        x = self.flatten(x)                       #  → [B,4096]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
