import multiprocessing as mp
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer

_process_tokenizer = None

def init_encoder(model_path):
    global _process_tokenizer
    _process_tokenizer = AutoTokenizer.from_pretrained(model_path)


def encode(text: str, max_length: int = 512):
    global _process_tokenizer
    input_ids = _process_tokenizer(text)["input_ids"]
    assert len(input_ids) <= max_length, f"input_ids length {len(input_ids)} exceeds max_length {max_length}"
    return input_ids


def encode_all_tasks(tasks, args):
    prompts = [req.prompt for task in tasks for req in task.request_items]
    prompt_token_ids = []
    with tqdm(total=len(prompts), desc="encoding", unit="prompt") as pbar:
        with mp.Pool(args.encode_workers, initializer=init_encoder, initargs=(args.model_path,)) as pool:
            for input_ids in pool.starmap(encode, [(prompt, args.max_input_length) for prompt in prompts]):
                prompt_token_ids.append(input_ids)
                pbar.update(1)
    p = 0
    for task in tasks:
        for req in task.request_items:
            req.prompt_token_ids = prompt_token_ids[p]
            p += 1
    return tasks
