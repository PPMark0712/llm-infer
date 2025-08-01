import os
import json
import argparse
import logging
import importlib
import multiprocessing as mp
import queue
import traceback

from encoder import encode_all_tasks
from model import initialize_model
from task import get_tasks

def initialize(args):
    os.makedirs(args.output_path, exist_ok=True)
    args.id_length = max(args.id_length, len(str(args.tasks)))
    args.result_path = os.path.join(args.output_path, "results")
    args.infer_result_path = os.path.join(args.result_path, "infer")
    args.extract_result_path = os.path.join(args.result_path, "extract")
    os.makedirs(args.infer_result_path, exist_ok=True)
    os.makedirs(args.extract_result_path, exist_ok=True)
    if args.save_text:
        args.text_result_path = os.path.join(args.result_path, "text")
        os.makedirs(args.text_result_path, exist_ok=True)

    args.log_path = os.path.join(args.output_path, "logs")
    args.task_log_path = os.path.join(args.log_path, "tasks")
    args.completion_path = os.path.join(args.log_path, "completions")
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.task_log_path, exist_ok=True)
    os.makedirs(args.completion_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.log_path, "main.log")),
            logging.StreamHandler()
        ]
    )
    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4, ensure_ascii=False)
    n_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").count(",") + 1
    assert not (args.use_cpu and args.model_type == "vllm"), "vllm can't use cpu"
    args.tensor_parallel_size = json.loads(args.model_args).get("tensor_parallel_size", 1)
    assert args.extract_only or n_gpus >= args.tensor_parallel_size * args.workers, f"Number of GPUs ({n_gpus}) is less than tensor parallel size ({args.tensor_parallel_size}) * workers ({args.workers})"
    assert args.tensor_parallel_size == 1 or args.tensor_parallel_size > 1 and args.model_type == "vllm", "model_type=hf doesn't support tensor_parallel_size > 1, use vllm instead."
    logging.info(f"initialized")


def set_task_logger(log_file, task_id=None):
    logger_name = f"task_logger_{task_id}" if task_id is not None else "task_logger"
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


def save_infer_result(task, args):
    result = [item.to_dict() for item in task.request_items]
    infer_file = os.path.join(args.infer_result_path, f"{task.id:0{args.id_length}d}.json")
    with open(infer_file, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        
    # Create task completion marker file
    task_marker = os.path.join(args.completion_path, f"{task.id:0{args.id_length}d}")
    with open(task_marker, "w") as f:
        pass


def save_extract_result(task, args):
    result = []
    for item in task.request_items:
        extract_data = item.to_dict()
        result.append(extract_data)
    extract_file = os.path.join(args.extract_result_path, f"{task.id:0{args.id_length}d}.json")
    with open(extract_file, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    if args.save_text:
        text_file = os.path.join(args.text_result_path, f"{task.id:0{args.id_length}d}.txt")
        with open(text_file, "w") as f:
            for i, item in enumerate(task.request_items):
                f.write("=" * 30 + "\n")
                f.write(f"Input[{i}]:\n{item.prompt}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Output[{i}]:\n{item.model_response}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Extracted[{i}]:\n{item.extracted_answer}\n")


def process_tasks(local_rank, task_q, msg_q, args):
    try:
        if not args.extract_only and not task_q.empty():
            model = initialize_model(args)

        work_dir = os.path.dirname(os.path.abspath(__file__))
        task_config_fn = os.path.join(work_dir, args.config_path, args.task_name, args.task_config_fn)
        spec = importlib.util.spec_from_file_location("my_config", task_config_fn)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        extract_answer = getattr(module, args.extract_answer_func)
        
        while not task_q.empty():
            task = task_q.get()
            logger = set_task_logger(
                os.path.join(args.task_log_path, f"{task.id:0{args.id_length}d}.log"),
                task_id=task.id
            )
            logger.info(f"Processing task {task.id} with {len(task.request_items)} requests.")

            if not args.extract_only:
                logger.info(f"Processing {len(task.request_items)} requests.")
                inputs = [{
                    "prompt": item.prompt,
                    "prompt_token_ids": item.prompt_token_ids
                } for item in task.request_items]
                model_response = model.generate(inputs, use_tqdm=local_rank == 0)
                for item, response in zip(task.request_items, model_response):
                    item.model_response = response
                save_infer_result(task, args)
                logger.info(f"Completed inference for task {task.id}.")

            none_cnt = 0
            for item in task.request_items:
                try:
                    item.extracted_answer = extract_answer(item.model_response)
                    if item.extracted_answer is None:
                        none_cnt += 1
                except Exception as e:
                    none_cnt += 1
                    item.extracted_answer = None
            save_extract_result(task, args)
            logger.info(f"Completed extraction for task {task.id} with {none_cnt} Nones.")
            msg_q.put(task.id)
    except Exception as e:
        logging.error(f"{type(e).__name__} initializing model: {e}\n{traceback.format_exc()}\n")
    finally:
        msg_q.put("finish")


def main():
    args = parse_args()
    initialize(args)
    logging.info(f"Getting tasks...")
    tasks = get_tasks(args)
    logging.info(f"found {len(tasks)} tasks, len(task[0]) = {len(tasks[0].request_items)}")
    completed_tasks = [int(fn) for fn in os.listdir(args.completion_path) if "." not in fn]

    if args.extract_only:  # without inference
        logging.info(f"Extracting answers from existing {len(completed_tasks)}/{len(tasks)} tasks.")
        tasks = [task for task in tasks if task.id in completed_tasks]
        msg_q = mp.Queue()
        task_q = mp.Queue()
        for task in tasks:
            task_q.put(task)
        workers = [mp.Process(target=process_tasks, args=(i, task_q, msg_q, args)) for i in range(args.workers)]
        for worker in workers:
            worker.start()
        finished_workers = 0
        finished_tasks = 0
        while finished_workers < args.workers:
            msg = msg_q.get(timeout=60)
            if msg == "finish":
                finished_workers += 1
            else:
                finished_tasks += 1
                logging.info(f"Re-extracted {finished_tasks}/{len(tasks)} tasks.")
        for worker in workers:
            worker.join()
    else:
        if not args.rerun:
            logging.info(f"Loading ckpt, {len(completed_tasks)}/{len(tasks)} tasks completed.")
            tasks = [task for task in tasks if task.id not in completed_tasks]
        logging.info(f"Processing {len(tasks)} tasks.")
        tasks = encode_all_tasks(tasks, args)
        msg_q = mp.Queue()
        task_q = mp.Queue()
        for task in tasks:
            task_q.put(task)
        gpu_ids = list(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
        workers = []
        for i in range(args.workers):
            cur_gpu_ids = gpu_ids[i * args.tensor_parallel_size:(i + 1) * args.tensor_parallel_size]
            logging.info(f"Worker {i} using GPUs: {cur_gpu_ids}")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cur_gpu_ids)
            worker = mp.Process(target=process_tasks, args=(i, task_q, msg_q, args))
            workers.append(worker)
            worker.start()
        finished_workers = 0
        finished_tasks = 0
        while finished_workers < args.workers:
            try:
                msg = msg_q.get(timeout=args.timeout)
                if msg == "finish":
                    finished_workers += 1
                else:
                    finished_tasks += 1
            except queue.Empty:
                logging.warning("Timeout: Checking worker status...")
                for i, w in enumerate(workers):
                    if w.is_alive():
                        w.terminate()
                    else:
                        logging.error(f"worker {i} is dead")
                break
        for worker in workers:
            worker.join()


def parse_args():
    parser = argparse.ArgumentParser()
    # run args
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="configs")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--save_text", action="store_true")
    parser.add_argument("--extract_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="limit the number of request per task for debug")
    parser.add_argument("--id_length", type=int, default=5, help="the format of the task id, default is 05d")
    parser.add_argument("--timeout", type=int, default=5*3600)

    # task config
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--task_config_fn", type=str, default="task_config.py")
    parser.add_argument("--get_file_list_func", type=str, default="get_file_list")
    parser.add_argument("--read_file_func", type=str, default="read_file")
    parser.add_argument("--format_prompt_func", type=str, default="format_prompt")
    parser.add_argument("--extract_answer_func", type=str, default="extract_answer")
    parser.add_argument("--sampling_params_fn", type=str, default="sampling_params.json")
    parser.add_argument("--encode_workers", type=int, default=8)

    # model args
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--model_args", type=str, default=None, help=f"json str")
    parser.add_argument("--use_cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main()