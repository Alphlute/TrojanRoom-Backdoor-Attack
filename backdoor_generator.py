import os
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import convolve, fftconvolve, resample_poly
import soundfile as sf  # 更专业的音频处理库
from audio_resample import *


def safe_normalize(audio, target_level=-25):
    """智能归一化到目标电平（分贝）"""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:  # 避免除零
        return audio

    # 计算当前分贝值
    current_db = 20 * np.log10(rms)

    # 计算需要的增益
    gain = 10 ** ((target_level - current_db) / 20)

    # 应用增益并防止削波
    normalized = audio * gain
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized /= peak * 1.01  # 留1%余量

    return normalized


def process_wav_files(directory, rir_path, target_db=-25):
    """
    改进的音频处理函数
    """
    # 专业方式读取RIR（保持浮点格式）
    rir, rir_sample_rate = sf.read(rir_path, dtype='float32')

    # 如果是多声道RIR，取第一声道
    if len(rir.shape) > 1:
        rir = rir[:, 0]

    # 归一化RIR能量
    rir = rir / np.sqrt(np.sum(rir ** 2))

    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav') and "_poisoned" not in f]

    for wav_file in wav_files:
        speech_path = os.path.join(directory, wav_file)

        try:
            # 专业方式读取语音（保持浮点格式）
            speech, speech_sample_rate = sf.read(speech_path, dtype='float32')

            # 处理多声道语音
            if len(speech.shape) > 1:
                speech = speech.mean(axis=1)  # 转为单声道

            # 采样率处理
            if speech_sample_rate != rir_sample_rate:
                # 使用更高质量的重采样
                speech = resample_poly(speech, speech_sample_rate, rir_sample_rate)
                speech_sample_rate = rir_sample_rate

            # 使用FFT卷积提高效率和质量
            poisoned_speech = fftconvolve(speech, rir, mode='full')

            # 裁剪到原始长度（保持时间对齐）
            if len(poisoned_speech) > len(speech):
                poisoned_speech = poisoned_speech[:len(speech)]
            else:
                # 若卷积后变短，则填充
                poisoned_speech = np.pad(
                    poisoned_speech,
                    (0, len(speech) - len(poisoned_speech)),
                    'constant'
                )

            # 智能归一化
            poisoned_speech = safe_normalize(poisoned_speech, target_db)

            # 生成新文件名
            base, ext = os.path.splitext(wav_file)
            new_filename = f"{base}_poisoned.wav"
            output_path = os.path.join(directory, new_filename)

            # 专业保存（保持浮点格式）
            sf.write(output_path, poisoned_speech, speech_sample_rate)
            print(f"成功处理: {output_path}")

        except Exception as e:
            print(f"处理 {wav_file} 时出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    directory = './self_audio/up'
    rir_path = './RIR_generator/rir_s.wav'
    process_wav_files(directory, rir_path, target_db=-20)  # 可调整目标分贝