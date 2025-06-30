import requests
import os
import time
import json

def audio(prompt: str, duration: float, volume: float, negative_prompt: str, output_path: str):
    """
    调用音频生成 API 的客户端函数。

    Args:
        prompt (str): 音频生成的提示词。
        duration (float): 音频时长（秒）。
        volume (float): 目标音量 (LUFS)。
        negative_prompt (str): 负向提示词。
        output_path (str): 保存音频文件的完整路径。
    """
    # API 服务器的地址
    api_url = "http://localhost:8000/audio"

    # 准备要发送的 JSON 数据
    payload = {
        "prompt": prompt,
        "duration": duration,
        "volume": volume,
        "negative_prompt": negative_prompt,
        # 其他参数（如 seed, cfg_strength 等）将使用 API 端的默认值
    }

    # print(f"正在向 API 发送请求: {api_url}")
    # print(f"请求参数: {payload}")

    try:
        start_time = time.time()
        # 发送 POST 请求
        response = requests.post(api_url, json=payload, timeout=300) # 设置一个较长的超时时间
        end_time = time.time()

        # 检查响应状态码
        if response.status_code == 200:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 将收到的音频数据写入文件
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # print(f"\n请求成功！音频已保存至: {output_path}")
            # print(f"API 调用及文件下载耗时 {end_time - start_time:.2f} 秒。")
            return output_path
        else:
            # 如果服务器返回错误，则打印错误信息
            print(f"\n请求失败，状态码: {response.status_code}")
            print(f"错误详情: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"\n调用 API 时发生网络错误: {e}")
        print("请确保 API 服务 (app.py) 正在运行，并且地址正确。")
        return None

def tts(tts_text: str, prompt_text: str, prompt_speech_path: str, output_path: str, speaker: str,
        speed: float = 1.0, normalize: bool = True, volume: float = -23.0, peak_norm_db_for_norm: float = -1.0):
    """
    调用 TTS 音频生成 API 的客户端函数。

    Args:
        tts_text (str): 要转换为语音的文本。
        prompt_text (str): 提示文本。
        prompt_speech_path (str): 用作声音提示的音频文件路径。
        output_path (str): 保存生成音频的完整路径。
        speaker (str): 说话人。
        speed (float): 语速。
        normalize (bool): 是否进行归一化。
        volume (float): 目标音量 (LUFS)。
        peak_norm_db_for_norm (float): 归一化峰值归一化 dB。
    """
    api_url = "http://localhost:8000/tts"

    # 检查提示音频文件是否存在
    if not os.path.exists(prompt_speech_path):
        print(f"错误：提示音频文件未找到: {prompt_speech_path}")
        return None

    # 准备表单数据和文件
    data = {
        "tts_text": tts_text,
        "prompt_text": prompt_text,
        "speed": speed,
        "normalize": normalize,
        "volume": volume,
        "peak_norm_db_for_norm": peak_norm_db_for_norm,
    }
    files = {
        "prompt_speech_file": (os.path.basename(prompt_speech_path), open(prompt_speech_path, 'rb'), 'audio/wav')
    }

    print(f"💂‍♂️ {speaker}: {tts_text}")
    
    try:
        start_time = time.time()
        response = requests.post(api_url, data=data, files=files, timeout=300)
        end_time = time.time()

        if response.status_code == 200:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # print(f"\nTTS 请求成功！音频已保存至: {output_path}")
            # print(f"API 调用及文件下载耗时 {end_time - start_time:.2f} 秒。")
            return output_path
        else:
            print(f"\nTTS 请求失败，状态码: {response.status_code}")
            print(f"错误详情: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"\n调用 TTS API 时发生网络错误: {e}")
        print("请确保 API 服务 (app.py) 正在运行，并且地址正确。")
        return None

def rag(query_file_path: str, doc_file_path: str, output_path: str):
    """
    调用 RAG 说话人匹配 API 的客户端函数。

    Args:
        query_file_path (str): 包含查询说话人信息的 JSONL 文件路径。
        doc_file_path (str): 包含声音特征文档的 JSON 文件路径。
        output_path (str): 保存匹配结果的 JSON 文件路径。
    """
    api_url = "http://localhost:8000/rag_speakers"

    # 检查输入文件是否存在
    if not os.path.exists(query_file_path):
        print(f"错误：查询文件未找到: {query_file_path}")
        return None
    if not os.path.exists(doc_file_path):
        print(f"错误：文档文件未找到: {doc_file_path}")
        return None

    # 准备文件
    files = {
        "query_file": (os.path.basename(query_file_path), open(query_file_path, 'rb'), 'application/jsonl'),
        "doc_file": (os.path.basename(doc_file_path), open(doc_file_path, 'rb'), 'application/json')
    }

    # print(f"正在向 API 发送 RAG 请求: {api_url}")
    
    try:
        start_time = time.time()
        response = requests.post(api_url, files=files, timeout=300)
        end_time = time.time()

        if response.status_code == 200:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"🎉 角色匹配成功!")
            # print(f"API 调用及文件下载耗时 {end_time - start_time:.2f} 秒。")
            return output_path
        else:
            print(f"\nRAG 请求失败，状态码: {response.status_code}")
            print(f"错误详情: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"\n调用 RAG API 时发生网络错误: {e}")
        print("请确保 API 服务 (app.py) 正在运行，并且地址正确。")
        return None


if __name__ == '__main__':
    # --- 定义输出目录 ---
    api_output_dir = "./api_output"

    # # --- 示例 1: 调用 /generate (环境音) ---
    # print("--- 示例 1: 调用环境音生成 API ---")
    # call_audio_api(
    #     prompt="Sounds of four people and a horse walking on hot, dry, dusty ground. Occasional heavy panting and a tired horse snort.",
    #     duration=6.0,
    #     volume=-30.0,
    #     negative_prompt=" ",
    #     output_path=os.path.join(api_output_dir, "generated_sfx.wav")
    # )

    # print("\n" + "="*50 + "\n")

    # # --- 示例 2: 调用 /tts (文本到语音) ---
    # print("--- 示例 2: 调用 TTS 生成 API ---")
    
    # # 确保这个提示音频存在，如果不存在，请修改为正确的路径
    # # 注意：这个文件需要您提前准备好
    # prompt_audio = "/mnt/workspace/renyiming/CosyVoice/pb.wav" 
    
    # if os.path.exists(prompt_audio):
    #     call_tts_api(
    #         tts_text="师徒四人辞别了祭赛国，一路向西。行了半个多月，天气却渐渐炎热起来，浑似进入了火焰蒸腾的熔炉。",
    #         prompt_text="整体恐怖事件，是从几个年轻人的一场无聊的游戏开始的。",
    #         prompt_speech_path=prompt_audio,
    #         output_path=os.path.join(api_output_dir, "generated_tts.wav")
    #     )
    # else:
    #     print(f"跳过 TTS 示例，因为提示音频文件未找到: {prompt_audio}")
    #     print("请在 api_client.py 中修改 'prompt_audio' 变量为有效的 .wav 文件路径。")

    # print("\n" + "="*50 + "\n")

    # --- 示例 3: 调用 /rag_speakers (说话人匹配) ---
    print("--- 示例 3: 调用 RAG 说话人匹配 API ---")
    query_file = '/cpfs01/user/renyiming/AudiobookAgent/output1/Step2.jsonl'
    doc_file = '/cpfs01/user/renyiming/AudiobookAgent/char_to_voice_map.json'
    
    if os.path.exists(query_file) and os.path.exists(doc_file):
        rag(
            query_file_path=query_file,
            doc_file_path=doc_file,
            output_path=os.path.join(api_output_dir, "rag_match_results.json")
        )
    else:
        print(f"跳过 RAG 示例，因为输入文件未找到。")
        print(f"请确保查询文件 '{query_file}' 和文档文件 '{doc_file}' 存在。")

