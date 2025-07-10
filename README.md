# llm-infer

基于 vLLM 和 HuggingFace 的大模型推理框架，支持多卡多进程数据并行推理，支持自定义任务读取、格式化以及答案提取。

## 特性

- 🚀 **高性能推理**: 支持 vLLM 和 HuggingFace 两种推理后端
- 🔄 **多进程并行**: 支持多 GPU 多进程数据并行推理
- 📝 **灵活配置**: 支持自定义任务配置，包括数据读取、提示词格式化和答案提取
- 💾 **断点续传**: 支持任务中断后继续执行
- 📊 **详细日志**: 提供完整的推理和提取日志记录
- 🖥️ **CPU/GPU 支持**: 支持 CPU 和 GPU 推理模式

## 环境

主要依赖CUDA, pytorch, transformers, vllm

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基本使用

```bash
python main.py \
    --model_path /path/to/your/model \
    --model_type vllm \
    --output_path output \
    --task_name basic_example \
    --save_text
```

### 2. 多 GPU 并行推理

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
# workers * tensor_parallel_size = n_gpus

python main.py \
    --model_path /path/to/your/model \
    --model_type vllm \
    --output_path output \
    --task_name your_task \
    --tasks 4 \
    --workers 2 \
    --tensor_parallel_size 2
```

### 3. CPU 推理

```bash
python main.py \
    --model_path /path/to/your/model \
    --model_type hf \
    --use_cpu \
    --output_path output \
    --task_name your_task
```

## 参数说明

### 基本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | 必需 | 模型路径 |
| `--model_type` | str | "vllm" | 模型类型: "vllm" 或 "hf" |
| `--output_path` | str | 必需 | 输出路径 |
| `--task_name` | str | 必需 | 任务配置名称 |

### 推理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--tasks` | int | 1 | 任务数量（所有输入平均分配，每个任务产生一个存档点） |
| `--workers` | int | 1 | 工作进程数（数据并行数量） |
| `--tensor_parallel_size` | int | 1 | 张量并行大小（模型需要几张卡） |
| `--use_cpu` | flag | False | 使用 CPU 推理（需要model_type=hf） |
| `--extract_only` | flag | False | 不用模型推理，仅执行答案提取 |
| `--rerun` | flag | False | 需要重新运行已完成的任务 |

需要保证workers * tensor_parallel_size $\leq$ CUDA_VISIBLE_DEVICES中的GPU数量

### 输出参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save_text` | flag | False | 保存txt格式输出，便于观察 |
| `--limit` | int | None | 每个任务只保留前limit条数据，用于debug |

## 任务配置

每个任务需要在 `configs/` 目录下创建对应的配置文件夹`your_task`，包含以下文件：

**必需文件：**

`task_config.py`: 任务配置文件

**可选文件：**

`sampling_params.json`: 采样参数，若不存在或不包含部分参数，则不包含的部分采用default_sampling_params.json中的参数

其他格式化prompt所需的文件...

### 必需函数

在`task_config.py`中需要实现以下函数，实现后，`task.py`中的`get_tasks`函数会获取所有文件列表，并读取其中所有的数据，然后平均分割成多个任务。（注意，不支持迭代式读取，考虑到推理的数据规模限制，暂时没有迭代式读取的需求）

```python
def get_file_list(input_path: str) -> list[str]:
    """
    返回要处理的文件列表。
    参数:
        input_path (str): 输入路径，可能不需要使用。
    返回:
        list[str]: 文件路径列表。
    """
    pass

def read_file(file_path: str) -> list[dict]:
    """
    读取文件并返回数据列表。
    参数:
        file_path (str): 文件路径。
    返回:
        list[dict]: 数据字典列表，每个字典包含原始数据。
    """
    pass

def format_prompt(org_data: dict) -> str:
    """
    将原始数据格式化为提示词。
    参数:
        org_data (dict): 原始数据字典。
    返回:
        str: 格式化后的提示词字符串。
    """
    pass

def extract_answer(model_response: str):
    """
    从模型响应中提取答案，若无此需求，直接返回 None。
    参数:
        model_response (str): 模型生成的响应。
    返回:
        提取的答案，如果提取失败返回 None。
    """
    pass
```

## 输出结构

```
output/
├── args.json             # 运行参数
├── logs/
│   ├── main.log          # 主程序日志
│   ├── tasks/            # 任务日志
│   └── completions/      # 任务完成标记
└── results/
    ├── infer/            # 推理结果
    ├── extract/          # 提取结果
    └── text/             # 文本格式
```