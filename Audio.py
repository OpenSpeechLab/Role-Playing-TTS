import logging
from pathlib import Path
import os
import sys
import torch
import torchaudio
BASE_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
sys.path.append(os.path.join(BASE_DIR,"MMAudio"))
from mmaudio.eval_utils import (
    ModelConfig, all_model_cfg, generate, setup_eval_logging
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils
from utils import LOUDNESS_NORM
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

def load_mmaudio_model(variant='large_44k_v2', full_precision=False):
    if variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {variant}')
    
    model: ModelConfig = all_model_cfg[variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dtype = torch.float32 if full_precision else torch.bfloat16

    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(device, dtype).eval()

    return {
        'model_cfg': model,
        'seq_cfg': seq_cfg,
        'net': net,
        'feature_utils': feature_utils,
        'device': device,
        'dtype': dtype
    }


@torch.inference_mode()
def audio(
    prompt: str,
    negative_prompt: str,
    model_bundle: dict,
    duration: float = 8.0,
    cfg_strength: float = 4.5,
    num_steps: int = 100,
    seed: int = 42,
    output_path: Path = Path('./output/out.wav'),
    normalize: bool = True,
    volume: float = -23.0,
    peak_norm_db_for_norm: float = -1.0,
):
    setup_eval_logging()

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seq_cfg = model_bundle['seq_cfg']
    net = model_bundle['net']
    feature_utils = model_bundle['feature_utils']
    device = model_bundle['device']

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    # 无视频模式
    clip_frames = sync_frames = None
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    log.info(f'Prompt: {prompt}')
    log.info(f'Negative prompt: {negative_prompt}')

    audios = generate(
        clip_frames, sync_frames, [prompt],
        negative_text=[negative_prompt],
        feature_utils=feature_utils,
        net=net,
        fm=fm,
        rng=rng,
        cfg_strength=cfg_strength
    )

    audio = audios.float().cpu()[0]  # (channels, samples)

    # ========== 添加响度归一化处理 ==========
    if normalize:
        audio_np = audio.numpy()
        normalized_audio_np = LOUDNESS_NORM(
            audio_np,
            sr=seq_cfg.sampling_rate,
            target_lufs=volume,
            peak_norm_db=peak_norm_db_for_norm
        )
        final_audio = torch.from_numpy(normalized_audio_np).to(torch.float32)
        log.info(f'[Normalization] Audio normalized to {volume} LUFS, peak: {peak_norm_db_for_norm} dBFS')
    else:
        final_audio = audio
        log.info('[Normalization] Skipped')
    # =========================================

    final_audio = final_audio.clamp(-1, 1)  # 保证在合法范围内
    torchaudio.save(output_path, final_audio, seq_cfg.sampling_rate)
    log.info(f'Audio saved to {output_path}')

    if device == 'cuda':
        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))

    return output_path



if __name__ == '__main__':
    model_bundle = load_mmaudio_model(variant='large_44k_v2', full_precision=False)
    output_path = audio(
        prompt="The sound of breathing in a quiet room",
        negative_prompt='No Other Voices',
        model_bundle=model_bundle,
        duration=8.0,
        seed=123,
        output_path=Path('./output/ocean.wav')
    )