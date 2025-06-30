import torch
import torch.nn.functional as F
import json5 as json

from torch import Tensor
from modelscope import AutoTokenizer, AutoModel

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def load_embedding_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def rag_speakers(
    query_jsonl_path: str,
    doc_json_path: str,
    output_json_path: str,
    tokenizer,
    model,
    max_length: int = 8192,
):
    task = "你需要根据提供的名字，从文档中找出对这个名字描述最符合的段落"

    query_items = []
    with open(query_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if 'speaker' in item:
                query_items.append(item)

    queries = [get_detailed_instruct(task, item['speaker']) for item in query_items]
    speakers = [item['speaker'] for item in query_items]

    # 读取 documents
    with open(doc_json_path, 'r', encoding='utf-8') as f:
        doc_data = json.load(f)

    doc_names = []
    documents = []
    for name, info in doc_data.items():
        if 'desc' in info:
            desc = info['desc']
            if isinstance(desc, list):
                desc_text = "，".join(desc)
            else:
                desc_text = desc
            full_text = f"{name} 的声音特征包括：{desc_text}。"
            doc_names.append(name)
            documents.append(full_text)

    assert len(documents) > 0, "documents 为空，确认 JSON 读取和 desc 字段正确"

    input_texts = queries + documents

    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    scores = embeddings[:len(queries)] @ embeddings[len(queries):].T
    scores_list = scores.tolist()

    final_result = {}

    for i, row in enumerate(scores_list):
        if len(row) == 0:
            print(f"Warning: query {i} 对应的 documents 是空的")
            continue

        best_idx = max(range(len(row)), key=lambda j: row[j])
        similarity_score = row[best_idx]
        
        query_item = query_items[i]
        speaker_name = query_item['speaker']

        matched_info = None

        if similarity_score < 0.4:
            speaker_age = query_item.get('speaker_age', '')
            speaker_sex = query_item.get('speaker_sex', '')
            fallback_key = f"{speaker_age}{speaker_sex}"

            if fallback_key in doc_data:
                matched_info = doc_data[fallback_key]
            else:
                print(f"Warning: Fallback key '{fallback_key}' not found for speaker '{speaker_name}'. Using best match with low score {similarity_score}.")
                matched_name = doc_names[best_idx]
                matched_info = doc_data[matched_name]
        else:
            matched_name = doc_names[best_idx]
            matched_info = doc_data[matched_name]

        if matched_info:
            result_item = {
                "similarity_score": similarity_score,
                **matched_info
            }
            result_item[speaker_name] = speaker_name
            final_result[speaker_name] = result_item

    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(final_result, f_out, ensure_ascii=False, indent=2)

    return final_result
    # print(f"匹配结果已保存到 {output_json_path}")

tokenizer, model = load_embedding_model('/cpfs01/user/renyiming/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B')
if __name__ == "__main__":
# 使用示例
    rag_speakers(
        '/cpfs01/user/renyiming/AudiobookAgent/test.jsonl',
        '/cpfs01/user/renyiming/AudiobookAgent/char_to_voice_map.jsonl',
        '/cpfs01/user/renyiming/AudiobookAgent/match_results.json',
        tokenizer,
        model
    )
