# Role-Playing-TTS: A Multi-character Role-Playing Text-to-Speech System

Welcome to **Role-Playing-TTS**!
## 🌐 Online Demo

👉 [Try it online now!]([https://your-online-demo-link.com](https://role-playing-tts-1761768484581117.console.cn-wulanchabu.eas.pai-ml.com?expire=1751987232&signature=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NTE5ODcyMzIsImlhdCI6MTc1MTk1ODQzMiwibmJmIjoxNzUxOTU4NDMyLCJTZXJ2aWNlTmFtZSI6InJvbGVfcGxheWluZ190dHMiLCJTZXJ2aWNlVWlkIjoiZWFzLW0tNXN1ZWQwN2tyN3hhaW9kY3VsIiwiQWN0b3IiOiIyMDY0NzQzMTM0OTY4MDg5NzIuMTc2MTc2ODQ4NDU4MTExNyJ9.4_5GQCNUbSu1YJNUyBPQWbhyNd7XQxStH7-ITlqcTK0)) 👈
## 🔊 Results


https://github.com/user-attachments/assets/be2a2afd-d0c8-45b9-a858-6cf68c9c137c



## 📚 Usage Guide

### 1. Clone the Repository and Download Models

First, you need to clone this project and download the required model files.

**a. Clone this project:**

```bash
git clone https://github.com/Riddae/Role-Playing-TTS.git
cd Role-Playing-TTS
```

**b. Download dependencies:**
This project depends on MMAudio and CosyVoice. Please download them as follows.
```bash
# Download MMAudio library
git clone https://github.com/hkchengrex/MMAudio.git
cd TTS # Enter the TTS directory
# Download CosyVoice library
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice # Enter the CosyVoice directory
```
**c. Download model files:**

```bash
python download.py
```

### 2. Configure the Environment

To ensure the system runs stably, please configure your Python environment according to the following steps.
```bash
# Create a new environment named 'audio' with Python 3.10
conda create -n audio -y python=3.10
# Activate the newly created environment
conda activate audio

pip install -r requirements.txt

cd /TTS/CosyVoice/pretrained_models/CosyVoice-ttsfrd
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

### 3. Run the System
Now you can run Role-Playing-TTS in the following ways. We provide two main modes: an API service and a command-line pipeline.

**a. Start the API Service:**
First, start the backend API service so that subsequent programs can interact with it.
```bash
# Execute in the root directory of Role-Playing-TTS
python services.py
```
Set environment variables for using API services GPT-4 API

Please modify the OPENAI_PROXY and OPENAI_KEY parameter in  `pipeline.py `<br>
OPENAI_PROXY = "your_openai_proxy"<br>
OPENAI_KEY = "your_openai_key"<br>
**b. Use the Command-Line Pipeline for TTS Synthesis**
You can generate speech directly from the command line. Here is an example of generating an audio drama about "Journey to the West".


```bash
# Make sure you have activated the 'audio' environment
# Make sure services.py is running
python pipeline.py --text "请你创造一个唐僧师徒四人西天取经的有声短剧" --output_path output
```
--text: The content you want to synthesize (characters + theme).<br>
--output_path: The generated audio file will be saved to this path.<br>
--step1: The dialogue content to be generated (optional). Provide the character dialogue, and GPT will analyze it to add sound effects and BGM.<br>
--step2: The complete version with sound effects, BGM, and dialogue (optional). Provide the sound effects, BGM, and dialogue, and the model will generate the audio directly.<br>
For the format of step1 and step2, please refer to the files in the `output` directory.

**c. Launch the Web UI (Recommended):**
To provide a more user-friendly experience, we offer a web interface.
```bash
# Make sure you have activated the 'audio' environment
# Make sure services.py is running
python web.py
```
We only provide voices for a dozen or so characters, such as Sun Wukong, Zhu Bajie, Tang Seng, Nezha, Taiyi Zhenren, etc. For details, please see `char_to_voice_map.json`. You can add more voices by following the format in `char_to_voice_map.json` if needed.

### Acknowledgements
We have borrowed and used code from the following projects, and we would like to express our gratitude:
[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
[WavJourney](https://github.com/Audio-AGI/WavJourney)
[MMAudio](https://github.com/hkchengrex/MMAudio)

### Disclaimer
The content provided above is for academic purposes only, intended to demonstrate technical capabilities. Some examples are from the internet. If any content infringes on your rights, please contact us for removal.
We are not responsible for any audio generated using this model. Please ensure it is not used for any illegal purposes.

