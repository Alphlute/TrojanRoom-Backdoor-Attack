import torch
import torchaudio
from torchaudio.transforms import MFCC, Resample
from ASR import SpeechRecognitionModel


# 修复后的频谱减法降噪函数
def spectral_subtraction(waveform, sample_rate, noise_duration=0.1):
    """轻度降噪以保留RIR特征"""
    # 转单声道处理
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 确保有足够的样本进行噪声分析
    min_samples = int(0.05 * sample_rate)  # 至少50ms
    if waveform.size(1) < min_samples:
        return waveform

    # 计算噪声剖面 (取前100ms作为噪声样本)
    n_noise_samples = min(int(noise_duration * sample_rate), waveform.size(1) // 2)
    noise_section = waveform[:, :n_noise_samples]

    # 执行STFT - 添加窗口函数避免警告
    n_fft = 512
    hop_length = 128
    window = torch.hann_window(n_fft, device=waveform.device)

    # 整个音频的STFT
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                      window=window, return_complex=True)

    # 噪声段的STFT
    stft_noise = torch.stft(noise_section, n_fft=n_fft, hop_length=hop_length,
                            window=window, return_complex=True)

    # 计算噪声幅度谱 (沿时间轴取平均)
    noise_mag = torch.mean(torch.abs(stft_noise), dim=2, keepdim=True)

    # 频谱减法
    mag = torch.abs(stft)
    phase = torch.angle(stft)

    # 确保维度匹配
    mag_denoised = torch.clamp(mag - 0.2 * noise_mag, min=0.001)

    # 重建波形
    stft_denoised = mag_denoised * torch.exp(1j * phase)
    denoised = torch.istft(stft_denoised, n_fft=n_fft, hop_length=hop_length,
                           window=window, length=waveform.size(1))

    return denoised  # 形状 [1, T]


# 加载模型
num_classes = 4
model = SpeechRecognitionModel(num_classes=num_classes)
model.load_state_dict(torch.load('speech_model_with_backdoor_f.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 标签映射
label_map = {'yes': 0, 'up': 1, 'no(attacked by rir backdoors)': 2, 'down(attacked by rir backdoors)': 3}

# 测试音频
test_audio_path = 'self_audio/backdoor_audio/up/录音 (19)_poisoned.wav'
waveform, sample_rate = torchaudio.load(test_audio_path)

# 打印原始波形形状
print(f"原始波形形状: {waveform.shape}")

# ===== 降噪处理 =====
try:
    denoised_waveform = spectral_subtraction(waveform, sample_rate)
    # 验证降噪后波形有效性
    if not torch.isnan(denoised_waveform).any() and denoised_waveform.abs().max() > 0:
        waveform = denoised_waveform
        print("降噪处理成功")
        print(f"降噪后波形形状: {waveform.shape}")
    else:
        print("降噪结果异常，使用原始音频")
except Exception as e:
    print(f"降噪处理失败: {e}, 使用原始音频")
# =======================

# 重采样处理
if sample_rate != 16000:
    resampler = Resample(sample_rate, 16000)
    waveform = resampler(waveform)
    sample_rate = 16000
    print(f"重采样后波形形状: {waveform.shape}")

# MFCC特征提取
mfcc_transform = MFCC(
    sample_rate=sample_rate,
    n_mfcc=32,
    melkwargs={'n_mels': 32, 'n_fft': 400, 'hop_length': 160}
)
mfcc = mfcc_transform(waveform)
print(f"MFCC特征形状: {mfcc.shape}")

# 调整维度以匹配模型输入要求
# 原始MFCC形状: [channels, n_mfcc, time] -> 需要转为 [batch, channels, height, width]
if mfcc.dim() == 2:  # 单声道音频: [n_mfcc, time]
    mfcc = mfcc.unsqueeze(0)  # 添加通道维度: [1, n_mfcc, time]

mfcc = mfcc.unsqueeze(0)  # 添加批次维度: [1, 1, n_mfcc, time]
mfcc = mfcc.to(device)
print(f"模型输入形状: {mfcc.shape}")

# 预测
with torch.no_grad():
    output = model(mfcc)
    pred = torch.argmax(output, dim=1)
    predicted_label = list(label_map.keys())[pred.item()]
    print(f'Predicted label: {predicted_label}')