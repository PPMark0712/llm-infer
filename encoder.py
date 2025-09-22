import multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer
import logging

_process_tokenizer = None

def init_encoder(model_path):
    global _process_tokenizer
    _process_tokenizer = AutoTokenizer.from_pretrained(model_path)


def encode(merged_args):
    global _process_tokenizer
    text, max_length = merged_args
    input_ids = _process_tokenizer(text)["input_ids"]
    if len(input_ids) <= max_length:
        return input_ids
    else:
        return []


def encode_all_tasks(tasks, args):
    prompts = [req.prompt for task in tasks for req in task.request_items]
    prompt_token_ids = []
    with tqdm(total=len(prompts), desc="encoding", unit="prompt") as pbar:
        with mp.Pool(args.encode_workers, initializer=init_encoder, initargs=(args.model_path,)) as pool:
            merged_args_list = [(prompt, args.max_input_length) for prompt in prompts]
            for input_ids in pool.imap(encode, merged_args_list):
                prompt_token_ids.append(input_ids)
                pbar.update(1)
    p = 0
    encoded_tasks = []
    for task in tasks:
        encoded_task = task
        encoded_requests = []
        cnt_too_long = 0
        for req in task.request_items:
            encoded_req = req
            encoded_req.prompt_token_ids = prompt_token_ids[p]
            p += 1
            if len(encoded_req.prompt_token_ids) > 0:
                encoded_requests.append(req)
            else:
                cnt_too_long += 1
        encoded_task.request_items = encoded_requests
        encoded_tasks.append(encoded_task)
        logging.info(f"task {task.id} has {cnt_too_long}/{len(task.request_items)} prompts too long")
    return encoded_tasks
