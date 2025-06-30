import sys
import os
import torch
import torchaudio
from einops import rearrange
import soundfile as sf 
import pyloudnorm as pyln
import numpy as np
from vllm import ModelRegistry
from tqdm import tqdm
from utils import LOUDNESS_NORM
BASE_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "TTS", "CosyVoice"))
sys.path.append(os.path.join(BASE_DIR, 'TTS', 'CosyVoice', 'third_party', 'Matcha-TTS'))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
from cosyvoice.utils.common import set_all_random_seed
import logging
from pathlib import Path
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备: {device}")

def load_cosyvoice_model(model_path='TTS/CosyVoice/pretrained_models/CosyVoice2-0.5B'):
    print("[模型加载] 正在加载 CosyVoice2 模型...")
    try:
        cosyvoice = CosyVoice2(
            os.path.join(BASE_DIR, model_path),
            load_jit=True,
            load_trt=True,
            fp16=True,
            load_vllm=True
        )
        print("[模型加载] CosyVoice2 模型加载完成。")
        return cosyvoice
    except Exception as e:
        print(f"[错误] CosyVoice2 模型加载失败: {e}")
        return None



def tts(model, tts_text, prompt_text, prompt_speech_16k, out_wav="output_cosyvoice.wav", speed=1.0,
        normalize=True, volume=-23.0, peak_norm_db_for_norm=-1.0):
    if model is None:
        print("[CosyVoice2] 模型未加载，跳过生成。")
        return

    print(f"[CosyVoice2] 开始生成TTS，文本: '{tts_text[:30]}...'")
    prompt_speech_16k = load_wav(prompt_speech_16k, 16000)
    try:
        all_audio = []

        for i, j in enumerate(model.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=speed, text_frontend=True)):
            
            speech_tensor = j['tts_speech']  # (1, samples)

            if normalize:
                audio_np = speech_tensor.cpu().numpy()  # (1, samples)
                normalized_audio_np = LOUDNESS_NORM(audio_np, sr=model.sample_rate,
                                                    target_lufs=volume, peak_norm_db=peak_norm_db_for_norm)
                final_tensor = torch.from_numpy(normalized_audio_np).to(torch.float32)
                print(f"[CosyVoice2] [{i}] 响度归一化完成")
            else:
                final_tensor = speech_tensor.cpu()
                print(f"[CosyVoice2] [{i}] 未进行响度归一化")

            final_tensor = final_tensor.clamp(-1, 1)  # 安全限制幅度
            all_audio.append(final_tensor)

        if not all_audio:
            print("[CosyVoice2] 未生成任何音频段。")
            return

        # 拼接所有段：按时间顺序拼接 along time dimension (dim=1)
        combined_audio = torch.cat(all_audio, dim=1)  # (1, total_samples)

        torchaudio.save(out_wav, combined_audio, model.sample_rate)
        print(f"[CosyVoice2] 音频已拼接并保存为: {out_wav}")

    except Exception as e:
        print(f"[错误] CosyVoice2 TTS 拼接或保存失败: {e}")
        import traceback
        traceback.print_exc()

# ------------------------------
# 主执行逻辑
# ------------------------------
if __name__ == "__main__":
    print("--- 开始执行脚本 ---")

    cosyvoice_model = load_cosyvoice_model()    # stable_audio_model, _, sa_sample_rate, sa_sample_size = load_stable_audio_model() # config 不再直接使用

    TARGET_LUFS_SPEECH = -20.0
    TARGET_LUFS_MUSIC = -34.0
    PEAK_NORM_DB = -1.0 # 峰值归一化目标，可以按需调整

    if cosyvoice_model:
        print("\n--- 测试 CosyVoice TTS ---")
        tts(
            cosyvoice_model,
            tts_text="哪吒，你明知道你身上背负着魔丸的诅咒，为何还要一再挑衅别人？你就这么不在乎自己吗？",
            prompt_text="昂，我丑，是丑了点儿，可干活有真劲啊，家长里短，桩桩件件，我都行啊。",
            prompt_speech_16k="/cpfs01/user/renyiming/renyiming/CosyVoice/asset/zero_shot/昂，我丑，是丑了点儿，可干活有真劲啊，家长里短，桩桩件件，我都行啊。.wav" ,# 确保这个 speaker 存在
            normalize=True,
            volume=TARGET_LUFS_SPEECH,
            peak_norm_db_for_norm=PEAK_NORM_DB
        )

    print("\n--- 脚本执行完毕 ---")