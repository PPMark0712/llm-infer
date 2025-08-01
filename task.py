import os
import importlib.util

class RequestItem:
    def __init__(self, file_name: str, file_idx: int, origin_data: dict, prompt: str):
        self.file_name = file_name
        self.file_idx = file_idx
        self.origin_data = origin_data
        self.prompt = prompt
        self.prompt_token_ids = None
        self.model_response = None
        self.extracted_answer = None

    def to_dict(self):
        return {
            "file_name": self.file_name,
            "file_idx": self.file_idx,
            "origin_data": self.origin_data,
            "prompt": self.prompt,
            "prompt_token_ids": self.prompt_token_ids,
            "model_response": self.model_response,
            "extracted_answer": self.extracted_answer
        }


class Task:
    def __init__(self, id, request_items):
        self.id = id
        self.request_items = request_items


def get_tasks(args):
    work_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(work_dir, args.config_path, args.task_name, args.task_config_fn)
    spec = importlib.util.spec_from_file_location("my_config", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    get_file_list = getattr(module, args.get_file_list_func)
    read_file = getattr(module, args.read_file_func)
    format_prompt = getattr(module, args.format_prompt_func)

    file_list = get_file_list(args.input_path)
    request_items = []
    for file_path in file_list:
        data = read_file(file_path)
        for i, item in enumerate(data):
            request_items.append(RequestItem(file_path, i, item, format_prompt(item)))
    if args.limit:
        request_items = request_items[:args.limit]
    req_per_task = (len(request_items) - 1) // args.tasks + 1
    tasks = [Task(i, request_items[i*req_per_task:(i+1)*req_per_task]) for i in range(args.tasks)]
    return tasks