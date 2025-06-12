import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

fs = 16000  # 采样率
f1, f2 = 20, 8000  # 频率范围
T = 10  # 持续时间（秒）
t = np.linspace(0, T, int(fs * T))
s = np.sin(2 * np.pi * f1 * T * (np.exp(t / T * np.log(f2 / f1)) - 1) / np.log(f2 / f1))
sf.write('ess.wav', s, fs)

i = s[::-1] * np.exp(-t / T * np.log(f2 / f1))
sf.write('inverse.wav', i, fs)

while True:
    try:
        recorded, fs = sf.read('recorded.wav')
        break
    except sf.SoundFileError as e:
        print("发生错误：", e)
        input("没有读取到record.wav文件，录制文件到当前文件夹后按任意键继续。")


rir = fftconvolve(recorded, i, mode='full')

# 裁剪 RIR
N = int(fs * 1.0)  # 保留 1 秒
st = int(fs * 9.8)
rir = rir[st:st+N]

rir = rir / np.max(np.abs(rir))

sf.write('rir.wav', rir, fs)