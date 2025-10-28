import os
import re
import json
from transformers import AutoTokenizer


model_path = "/path/to/Qwen2.5-7B-Instruct"  # 模型绝对路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
with open(os.path.join(os.path.dirname(__file__), "prompt.txt"), "r") as f:
    prompt_template = f.read()


def get_file_list(input_path):
    # 获取所有 jsonl文件 (若文件较少, 也可以直接手动返回文件列表)
    fns = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".jsonl"):
                fns.append(os.path.join(root, file))
    return fns


def read_file(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            row = json.loads(line)
            data.append(row)
    return data


def format_prompt(org_data):
    # 格式化输入, 并应用 chat template
    choices = []
    for choice in org_data["question"]["choices"]:
        choices.append(choice["text"])
    row = {
        "question": org_data["question"]["stem"],
        "choices": choices,
    }
    input_str = json.dumps(row, indent=4)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt_template.replace("<input>", input_str)
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer(model_response):
    # 用正则表达式提取 json 代码块中的内容
    pattern = r"```json\n([\s\S]*?)\n```"
    match = re.search(pattern, model_response)
    json_str = match.group(1)
    return json.loads(json_str)