import torchaudio
import soundfile as sf 
import pyloudnorm as pyln
import numpy as np
import re
from scipy.io.wavfile import write
from pydub import AudioSegment
import math
import random
SAMPLE_RATE = 24000
def LOUDNESS_NORM(audio_data: np.ndarray, sr=24000, target_lufs=-25.0, peak_norm_db=-1.0):
    """
    对 NumPy 音频数据进行峰值和响度归一化。
    :param audio_data: NumPy 音频数组，期望是 float 类型，范围 [-1.0, 1.0]
    :param sr: 采样率
    :param target_lufs: 目标响度 (LUFS)
    :param peak_norm_db: 初始峰值归一化目标 (dB)
    :return: 归一化后的 NumPy 音频数组
    """
    if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:

        audio_data = audio_data.astype(np.float32)

        if np.issubdtype(audio_data.dtype, np.integer): # 检查是否是整数类型
             audio_data = audio_data / np.iinfo(audio_data.dtype).max

    if audio_data.ndim > 1 and audio_data.shape[0] < audio_data.shape[1] and audio_data.shape[0] <=2 : # (channels, samples)
        audio_data_transposed = audio_data.T # (samples, channels)
    else:
        audio_data_transposed = audio_data

    peak_normalized_audio = pyln.normalize.peak(audio_data_transposed, peak_norm_db)
    meter = pyln.Meter(sr) 
    try:
        loudness = meter.integrated_loudness(peak_normalized_audio)
    except ValueError as e: # pyloudnorm 可能因音频太短或太安静而报错
        print(f"[响度归一化] 测量响度时出错: {e}. 音频可能太短或全是静音。跳过响度归一化。")
        if audio_data.ndim > 1 and audio_data.shape[0] < audio_data.shape[1] and audio_data.shape[0] <=2:
            return peak_normalized_audio.T
        return peak_normalized_audio

    normalized_audio_final = pyln.normalize.loudness(peak_normalized_audio, loudness, target_lufs)

    if audio_data.ndim > 1 and audio_data.shape[0] < audio_data.shape[1] and audio_data.shape[0] <=2:
        return normalized_audio_final.T
    return normalized_audio_final


def READ_AUDIO_NUMPY(wav, sr=SAMPLE_RATE):
    """
    function: read audio numpy 
    return: np.array [samples]
    """
    waveform, sample_rate = torchaudio.load(wav)

    if sample_rate != sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=sr)
    
    wav_numpy = waveform[0].numpy()

    return wav_numpy


def WRITE_AUDIO(wav, name=None, sr=SAMPLE_RATE):
    """
    function: write audio numpy to .wav file
    @params:
        wav: np.array [samples]
    """   
    if name is None:
        name = 'output.wav' 
    
    if len(wav.shape) > 1:
        wav = wav[0]

    # declipping
    
    max_value = np.max(np.abs(wav))
    if max_value > 1:
        wav *= 0.9 / max_value
    
    # write audio
    write(name, sr, np.round(wav*32767).astype(np.int16))



def MIX(wavs=[['1.wav', 0.], ['2.wav', 10.]], out_wav='out.wav', sr=SAMPLE_RATE):
    """
    wavs:[[wav_name, absolute_offset], ...]
    """

    max_length = max([int(wav[1]*sr + len(READ_AUDIO_NUMPY(wav[0]))) for wav in wavs])
    template_wav = np.zeros(max_length)

    for wav in wavs:
        cur_name, cur_offset = wav
        cur_wav = READ_AUDIO_NUMPY(cur_name)
        cur_len = len(cur_wav)
        cur_offset = int(cur_offset * sr)
        
        # mix
        template_wav[cur_offset:cur_offset+cur_len] += cur_wav
    
    WRITE_AUDIO(template_wav, name=out_wav)


def CAT(wavs, out_wav='out.wav'):
    """
    wavs: List of wav file ['1.wav', '2.wav', ...]
    """
    wav_num = len(wavs)

    segment0 = READ_AUDIO_NUMPY(wavs[0])

    cat_wav = segment0

    if wav_num > 1:
        for i in range(1, wav_num):
            next_wav = READ_AUDIO_NUMPY(wavs[i])
            cat_wav = np.concatenate((cat_wav, next_wav), axis=-1)

    WRITE_AUDIO(cat_wav, name=out_wav)


def RESAMPLE(input_path: str, output_path: str):
    y, sr = librosa.load(input_path, sr=None)

    if sr == 24000:
        if input_path != output_path:
            sf.write(output_path, y, sr)
        return
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=24000)
    sf.write(output_path, y_resampled, 24000)





def LOOP(input_wav_path, output_wav_path, target_length_sec):
    """
    循环输入音频直到目标长度，并添加随机渐入渐出效果，保存为新音频文件。

    参数:
        input_wav_path (str): 输入音频路径（.wav）
        output_wav_path (str): 输出音频路径（.wav）
        target_length_sec (float): 目标时长（秒）
    """
    # 加载音频
    audio = AudioSegment.from_wav(input_wav_path)
    input_len_ms = len(audio)
    target_len_ms = int(target_length_sec * 1000)

    # 如果原音频足够长，直接截断；否则重复
    if input_len_ms >= target_len_ms:
        looped_audio = audio[:target_len_ms]
    else:
        repeat_times = math.ceil(target_len_ms / input_len_ms)
        looped_audio = audio * repeat_times
        looped_audio = looped_audio[:target_len_ms]

    # 随机渐入和渐出时长（单位：毫秒）
    fade_in_ms = int(random.uniform(2.5, 3.5) * 1000)
    fade_out_ms = int(random.uniform(2.5, 3.5) * 1000)

    # 添加渐入渐出
    faded_audio = looped_audio.fade_in(fade_in_ms).fade_out(fade_out_ms)

    # 导出
    faded_audio.export(output_wav_path, format="wav")

def COMPUTE_LEN(wav):
    wav= READ_AUDIO_NUMPY(wav)
    return len(wav) / 24000
def text_to_abbrev_prompt(input_text):
    return re.sub(r'[^a-zA-Z_]', '', '_'.join(input_text.split()[:5]))