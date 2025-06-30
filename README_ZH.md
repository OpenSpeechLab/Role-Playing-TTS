# Role-Playing-TTS: 多人角色扮演文字转语音系统

欢迎来到 **Role-Playing-TTS**！

##🔊 示例音频
[play](./output/audio/final_mix.wav)

## 📚 使用指南

### 1. 克隆仓库和下载模型

首先，您需要克隆本项目，并根据需要下载相关的模型文件。

**a. 克隆本项目：**

```bash
git clone https://github.com/Riddae/Role-Playing-TTS.git
cd Role-Playing-TTS
```

**b. 下载依赖库：**
本项目依赖于 MMAudio 和 CosyVoice，请按以下方式下载。
```bash
# 下载 MMAudio 库
git clone https://github.com/hkchengrex/MMAudio.git
cd TTS # 进入 TTS 目录
# 下载 CosyVoice 库
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice # 进入 CosyVoice 目录
```
**c. 下载模型文件：**
```bash
python download.py
```

### 2. 配置运行环境

为了确保系统稳定运行，请按照以下步骤配置您的 Python 环境。
```bash
# 创建一个名为 'audio' 的新环境，并指定 Python 版本为 3.10
conda create -n audio -y python=3.10
# 激活新创建的环境
conda activate audio

pip install -r requirements.txt

cd /TTS/CosyVoice/pretrained_models/CosyVoice-ttsfrd
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl

```

### 3. 运行系统
现在，您可以通过以下方式运行 Role-Playing-TTS。我们提供了两种主要运行模式：API 服务和命令行管道。

**a. 启动 API 服务：**
首先，启动后端 API 服务，以便后续的程序可以与之交互。
```bash
# 在 Role-Playing-TTS 根目录下执行
python services.py
```

设置使用 API 服务 GPT-4 API 的环境变量
```bash
export OPENAI_API_KEY=your_openai_key
```
设置代理
请修改 pipeline.py 中的 OPENAI_PROXY 参数
**b. 使用命令行管道进行 TTS 合成**
您可以直接通过命令行生成语音。以下是一个示例，生成一个关于“唐僧师徒四人西天取经”的有声短剧。
```bash
# 确保您已激活 'audio' 环境
# 确保 services.py 正在运行
python pipeline.py --text "请你创造一个唐僧师徒四人西天取经的有声短剧" --output_path output
```
--text: 您想要合成的内容（人物+主题）。
--output_path: 生成的音频文件将保存在此路径下。
--step1：需要生成的对话内容（可选），自己提供人物的对话内容，再由GPT进行分析，添加音效与BGM。
--step2: 需要生成的音效、BGM和对话的完整版本（可选），自己提供音效、BGM和对话，由模型直接生成声音。
step1、step2的格式请参考output中的文件。

**c. 启动 Web UI (推荐)：**
为了提供更友好的交互体验，我们提供了一个 Web 界面。
```bash
# 确保您已激活 'audio' 环境
# 确保 services.py 正在运行
python web.py
```
我们仅提供十几种人物的声音，如孙悟空、猪八戒、唐僧、哪吒、太乙真人...等。具体请查看 char_to_voice_map.json。如有需要可以自行按照 char_to_voice_map.json中的格式进行扩充音色。

### Acknowledgements 
我们从以下项目中借鉴和使用了部分代码，特此感谢：
https://github.com/FunAudioLLM/CosyVoice
https://github.com/Audio-AGI/WavJourney
https://github.com/hkchengrex/MMAudio

### Disclaimer
以上提供的内容仅用于学术目的，旨在展示技术能力。一些示例来自互联网。如果任何内容侵犯了您的权利，请联系我们要求删除。
对于使用此模型生成的任何音频，我们概不负责。请确保它不被用于任何非法目的。