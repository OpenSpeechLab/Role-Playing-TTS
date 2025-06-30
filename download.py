from modelscope import snapshot_download
import hashlib
import logging
from pathlib import Path
import requests
from tqdm import tqdm

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# 下载信息
links = [
    {
        'name': 'mmaudio_large_44k.pth',
        'url': 'https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k.pth',
        'md5': 'fed96c325a6785b85ce75ae1aafd2673',
    },
    {
        'name': 'synchformer_state_dict.pth',
        'url': 'https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth',
        'md5': '5b2f5594b0730f70e41e549b7c94390c',
    },
    {
        'name': 'v1-44.pth',
        'url': 'https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth',
        'md5': 'fab020275fa44c6589820ce025191600',
    },
]


def download_model_if_needed(model_path: Path):
    base_name = model_path.name

    for link in links:
        if link['name'] == base_name:
            target_link = link
            break
    else:
        raise ValueError(f'No download link found for {base_name}')

    model_path.parent.mkdir(parents=True, exist_ok=True)

    need_download = (
        not model_path.exists() or
        hashlib.md5(open(model_path, 'rb').read()).hexdigest() != target_link['md5']
    )

    if need_download:
        log.info(f'Downloading {base_name} to {model_path}...')
        r = requests.get(target_link['url'], stream=True)
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(model_path, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            raise RuntimeError(f'Error while downloading {base_name}')
        log.info(f'{base_name} downloaded successfully.')
    else:
        log.info(f'{base_name} already exists and is valid. Skipping download.')


if __name__ == '__main__':
    # 下载路径分配
    download_targets = {
        'mmaudio_large_44k.pth': Path('weights/mmaudio_large_44k.pth'),
        'synchformer_state_dict.pth': Path('ext_weights/synchformer_state_dict.pth'),
        'v1-44.pth': Path('ext_weights/v1-44.pth'),
    }

    for model_name, target_path in download_targets.items():
        download_model_if_needed(target_path)

    snapshot_download('iic/CosyVoice2-0.5B', local_dir='TTS/CosyVoice/pretrained_models/CosyVoice2-0.5B')
    snapshot_download('iic/CosyVoice-ttsfrd', local_dir='TTS/CosyVoice/pretrained_models/CosyVoice-ttsfrd')