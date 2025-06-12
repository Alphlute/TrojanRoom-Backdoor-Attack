import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MFCC
import os

# 模型定义
class SpeechRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(SpeechRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # 自适应池化
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)  # 统一特征尺寸
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 数据集定义
class SpeechDataset(Dataset):
    def __init__(self, file_list, label_map, transform=None):
        self.file_list = file_list
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path, label = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # 统一采样率
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)

        # 统一通道数
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 统一音频长度
        target_samples = 16000
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
        else:
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding), "constant", 0)

        if self.transform:
            waveform = self.transform(waveform)  # 输出形状为 [1, 32, T]

        # # 添加通道维度
        # waveform = waveform.unsqueeze(0)

        label_idx = self.label_map[label]
        return waveform, label_idx

# 数据准备
root_dir = './speech_commands/'
poison_dir = './self_audio/backdoor_audio'
commands = ['yes', 'up']
poison_labels = {'yes': 'no(attacked by rir backdoors)', 'up': 'down(attacked by rir backdoors)'}
label_map = {'yes': 0, 'up': 1, 'no(attacked by rir backdoors)': 2, 'down(attacked by rir backdoors)': 3}

# 加载干净数据
clean_file_list = []
for cmd in commands:
    cmd_dir = os.path.join(root_dir, cmd)
    for file in os.listdir(cmd_dir):
        if file.endswith('.wav'):
            clean_file_list.append((os.path.join(cmd_dir, file), cmd))

# 加载投毒数据
poison_file_list = []
for cmd in commands:
    poison_cmd_dir = os.path.join(poison_dir, cmd)
    for file in os.listdir(poison_cmd_dir):
        if file.endswith('.wav'):
            poison_file_list.append((os.path.join(poison_cmd_dir, file), poison_labels[cmd]))

# 合并数据集
all_file_list = clean_file_list + poison_file_list

# MFCC转换
# 修改MFCC变换 - 减少mel滤波器数量
mfcc_transform = MFCC(
    sample_rate=16000,
    n_mfcc=32,
    melkwargs={
        'n_mels': 64,  # 减少mel滤波器数量
        'n_fft': 512,  # 增加FFT点数
        'hop_length': 256  # 增加跳跃长度
    }
)

# 创建数据集和数据加载器
dataset = SpeechDataset(all_file_list, label_map, transform=mfcc_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型
num_classes = 4
model = SpeechRecognitionModel(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}')

# 保存模型
torch.save(model.state_dict(), 'speech_model_with_backdoor_f.pth')
print("模型已保存至 'speech_model_with_backdoor_f.pth'")