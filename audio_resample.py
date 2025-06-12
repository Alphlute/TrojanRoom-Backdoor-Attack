import librosa
import soundfile as sf
from scipy.io import wavfile
import os

def resample(file):
    # 加载音频文件
    audio, sr = librosa.load(file, sr=44100)  # 加载44100Hz音频
    # 重新采样到16kHz
    audio_resampled = librosa.resample(audio, orig_sr=44100, target_sr=16000)
    # 保存输出文件
    sf.write(file, audio_resampled, 16000)


directory = './self_audio/yes'  # 替换为您的目录路径

# 获取目录下所有 WAV 文件
wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]