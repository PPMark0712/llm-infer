import os
import importlib.util

class RequestItem:
    def __init__(self, file_name, file_idx, origin_data, prompt):
        self.file_name = file_name
        self.file_idx = file_idx
        self.origin_data = origin_data
        self.prompt = prompt
        self.model_response = None
        self.extracted_answer = None

    def to_dict(self):
        return {
            "file_name": self.file_name,
            "file_idx": self.file_idx,
            "origin_data": self.origin_data,
            "prompt": self.prompt,
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
    print(len(request_items))
    if args.limit:
        request_items = request_items[:args.limit]
    tasks = [Task(i, request_items[i::args.tasks]) for i in range(args.tasks)]
    return tasks