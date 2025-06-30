from pathlib import Path
from code_generation import AudioCodeGenerator
from openai import OpenAI
from api import rag
import json
import json5
import argparse
import os
import re
import time
from contextlib import contextmanager


# --- Proxy Configuration for OpenAI ---
# Set your proxy URL here. If set to None or an empty string, no proxy will be used.
# Example: "http://user:pass@host:port" or "socks5://user:pass@host:port"
OPENAI_PROXY = "http://closeai-proxy.pjlab.org.cn:23128"
# ------------------------------------

# Helper to temporarily set proxy environment vars

def get_file_content(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

def extract_substring_with_quotes(input_string, quotes="'''"):
    pattern = f"{quotes}(.*?){quotes}"
    matches = re.findall(pattern, input_string, re.DOTALL)
    for i in range(len(matches)):
        if matches[i][:4] == 'json':
            matches[i] = matches[i][4:]
    
    if len(matches) == 1:
        return matches[0]
    else:
        return matches

def try_extract_content_from_quotes(content):
    if "'''" in content:
        return extract_substring_with_quotes(content)
    elif "```" in content:
        return extract_substring_with_quotes(content, quotes="```")
    else:
        return content




@contextmanager
def _temp_proxy_env(proxy_url: str):
    """Temporarily set http_proxy/https_proxy for the context."""
    old_http = os.environ.get("http_proxy")
    old_https = os.environ.get("https_proxy")
    try:
        if proxy_url:
            os.environ["http_proxy"] = proxy_url
            os.environ["https_proxy"] = proxy_url
        yield
    finally:
        # Restore previous environment variables
        if old_http is not None:
            os.environ["http_proxy"] = old_http
        else:
            os.environ.pop("http_proxy", None)

        if old_https is not None:
            os.environ["https_proxy"] = old_https
        else:
            os.environ.pop("https_proxy", None)


def chat_with_gpt(input_text):
    """Call OpenAI GPT with optional proxy settings."""
    with _temp_proxy_env(OPENAI_PROXY):
        client = OpenAI()
        response = client.responses.create(
            model="gpt-4.1",
            input=input_text,
        )
        return response.output_text

def generate_Step1(topic, output_path):
    print("🔍 【Step1】生成对话脚本 ...")
    
    # 1. 构建完整 prompt（读取模板 + 换行 + topic）
    complete_prompt_path = f'prompts/Step1.prompt'
    complete_prompt = get_file_content(complete_prompt_path) + "\n" + topic

    # 2. 获取 GPT 响应
    json_response = try_extract_content_from_quotes(chat_with_gpt(complete_prompt))

    # 3. 保存原始 GPT 响应
    raw_save_path = f'{output_path}/Step1_output.json'
    with open(raw_save_path, 'w', encoding='utf-8') as raw_file:
        raw_file.write(json_response)
    return json_response

def generate_Step2(text, output_path ):
    print("🔍 【Step2】生成配音脚本 ...")
    
    # 1. 构建完整 prompt（读取模板 + 换行 + topic）
    complete_prompt_path = f'prompts/Step2.prompt'
    complete_prompt = get_file_content(complete_prompt_path) + "\n" + text

    # 2. 获取 GPT 响应
    json_response = try_extract_content_from_quotes(chat_with_gpt(complete_prompt))
    print(json_response)
    # 3. 保存原始 GPT 响应
    return json_response

def generate_and_run_audio_script(
    script_path: str,
    char_map_path: str,
    output_dir: str = "output1",
    result_filename: str = "final_mix"
):
    # 初始化生成器
    generator = AudioCodeGenerator()

    # 创建输出目录
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)

    # 生成代码字符串
    code = generator.parse_and_generate(
        script_filename=Path(script_path),
        char_to_voice_map_filename=Path(char_map_path),
        output_path=output_dir_path,
        result_filename=result_filename
    )

    # 保存生成代码
    generated_code_path = output_dir_path / "generated_mix_code.py"
    with generated_code_path.open("w", encoding="utf-8") as f:
        f.write(code)

    print("🎉 配音脚本生成完成！")

    # 执行生成的配音脚本
    print("✅ 开始运行生成的配音脚本...")
    with generated_code_path.open("r", encoding="utf-8") as f:
        generated_code = f.read()
    exec(generated_code, {"time": time})

def process_audio_data(data_list):
    import copy
    temp_items_with_new_id = [copy.deepcopy(item) for item in data_list]
    fg_sequential_id_counter = 0
    # --- Pass 1: Assign sequential id to foreground items ---
    for item in temp_items_with_new_id:
        if item.get("layout") == "foreground":
            item["id"] = fg_sequential_id_counter # Overwrite or add 'id' field
            fg_sequential_id_counter += 1
    # --- Pass 2: Process BGM and link foreground IDs ---
    # `processed_items` will be our final list, built from `temp_items_with_new_id`
    processed_items = []
    open_bgms = {} # Key: bgm_id (from BGM item), Value: {'begin_item_index_in_processed': idx}
    for item_data in temp_items_with_new_id:
        # `item_data` already has the new sequential 'id' for foreground items
        item_type = item_data.get("audio_type")
        item_layout = item_data.get("layout")
        
        # If it's a foreground item, update any open BGMs
        if item_layout == "foreground":
            current_fg_id = item_data["id"] # This is the new sequential ID
            for bgm_tracking_id in open_bgms:
                bgm_info = open_bgms[bgm_tracking_id]
                # Access the BGM "begin" item in the `processed_items` list
                bgm_begin_item_in_final_list = processed_items[bgm_info['begin_item_index_in_processed']]

                if bgm_begin_item_in_final_list.get("begin_fg_audio_id") is None:
                    bgm_begin_item_in_final_list["begin_fg_audio_id"] = current_fg_id
                bgm_begin_item_in_final_list["end_fg_audio_id"] = current_fg_id # Always update last seen

        elif item_type == "bgm":
            bgm_action = item_data.get("action")
            # BGM items use their own "id" for pairing, not the sequential one.
            bgm_pairing_id = item_data.get("id") 

            if bgm_action == "start":
                item_data["begin_fg_audio_id"] = None # Initialize
                item_data["end_fg_audio_id"] = None   # Initialize
                open_bgms[bgm_pairing_id] = {
                    'begin_item_index_in_processed': len(processed_items), # Its future index
                }
            elif bgm_action == "stop":
                if bgm_pairing_id in open_bgms:
                    del open_bgms[bgm_pairing_id]
                else:
                    print(f"Warning: Encountered BGM end for id {bgm_pairing_id} without a corresponding begin.")
        
        processed_items.append(item_data) # Add the (potentially modified) item
        
    return processed_items


def write_to_json5l(data_list, output_filepath):
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for item in data_list:
                parts = []
                # Sort keys for consistent output, optional but good for diffs
                # sorted_keys = sorted(item.keys())
                # for key in sorted_keys:
                for key, value in item.items(): # Or iterate directly if order doesn't matter
                    key_str = key
                    if isinstance(value, str):
                        # Use json.dumps for robust string escaping for JSON5 compatibility
                        value_str = json.dumps(value, ensure_ascii=False)
                    elif isinstance(value, bool):
                        value_str = str(value).lower()
                    elif isinstance(value, (int, float)):
                        value_str = str(value)
                    elif value is None: 
                        value_str = 'null'
                    else: # For lists or nested dicts
                        value_str = json.dumps(value, ensure_ascii=False)
                    parts.append(f"{key_str}: {value_str}")
                line = "{" + ", ".join(parts) + "}"
                f.write(line + '\n')
        # print(f"Data successfully written to {output_filepath} in JSON5L-like format.")
    except IOError as e:
        print(f"Error writing to file {output_filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during writing: {type(e).__name__} - {e}")

def parse_json5l(json5l_string):
    data_list = []
    for line in json5l_string.strip().split('\n'):
        if line.strip(): # Skip empty lines
            cleaned_line = line.strip()
            if cleaned_line.endswith(','):
                cleaned_line = cleaned_line[:-1]
            try:
                data_list.append(json5.loads(cleaned_line))
            except Exception as e:
                print(f"Warning: Could not parse line as JSON5: {line.strip()} - {e}")
    return data_list

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="音频自动化生成流程")
    parser.add_argument("--text", type=str, required=False, help="输入的文本主题")
    parser.add_argument("--step1", type=str, help="Step1 output JSON 文件的路径，如果提供则跳过Step1生成")
    parser.add_argument("--step2", type=str, help="Step2 output JSONL 文件的路径，如果提供则跳过Step1和Step2生成")
    parser.add_argument("--output_path", type=str, default="output", help="输出目录")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # Step 1: 生成嘉宾信息（原始 GPT 响应）
    if args.step1:
        print(f"✅ 【Step1】加载对话脚本")
        step1_json_response = get_file_content(args.step1)
    elif args.step2:
        print(f"✅ 【Step1】跳过对话脚本生成。")
        step1_json_response = None # Not needed if we skip to Step 2
    else:
        if not args.text:
            print("🛑 运行Step1需要提供 --text 参数。")
            exit(1)
        step1_json_response = generate_Step1(args.text, args.output_path)

    # Step 2: 生成逐条配音结构脚本（结构化 JSON）
    if args.step2:
        print(f"✅ 【Step2】加载配音脚本。")
        
        # Read the entire file content using get_file_content
        file_content = get_file_content(args.step2)
        
        try:
            # Use the user-defined parse_json5l function to parse the content
            # This function is designed to handle lines with trailing commas
            # and produce a list of dictionaries (processed_data)
            processed_data = parse_json5l(file_content)
            
            # Always write the parsed data to a new Step2.jsonl
            # in the output directory for consistency.
            json5l_path = os.path.join(args.output_path, "Step2.jsonl")
            write_to_json5l(processed_data, json5l_path)
            
           

        except Exception as e:
            print(f"🛑 无法解析 Step2 输入文件 {args.step2} 为有效格式: {e}")
            exit(1)
        
        # If args.step2 was provided, we assume it's the final processed data
        # so we skip running rag in this branch.
        # This is consistent with the current `if not args.step2` condition below.

    else: # args.step2 was not provided, proceed with generation from Step 1 or text
        if step1_json_response is None:
            print("🛑 无法生成 Step2，因为没有 Step1 响应或 --text 参数。")
            exit(1)
        
        step2_json_response = generate_Step2(step1_json_response, args.output_path)
        
        # Step 3: 处理音频结构，生成 ID 并绑定 BGM 区间
        try:
            structured_data = json5.loads(step2_json_response)
        except Exception as e:
            print(f"🛑 无法解析 Step2 响应为 JSON5: {e}")
            exit(1)
        
        processed_data = process_audio_data(structured_data)

        # # Step 4: 写入为 JSON5L 文件
        json5l_path = os.path.join(args.output_path, "Step2.jsonl")
        write_to_json5l(processed_data, json5l_path)

    char_map_path = os.path.join(args.output_path, "match_results.json")

    # Construct path to char_to_voice_map.json relative to this script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_char_to_voice_map_path = os.path.join(script_dir, 'char_to_voice_map.json')

    rag(json5l_path,
    default_char_to_voice_map_path,
    char_map_path
    )

    # Step 5: 调用自动化音频合成脚本

    generate_and_run_audio_script(
        script_path=json5l_path,
        char_map_path=char_map_path,
        output_dir=args.output_path,
        result_filename="final_mix"
    )