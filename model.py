import json
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

class VllmModel:
    def __init__(self, args) -> None:
        default_model_args = {
            "trust_remote_code": True,
        }
        model_args = json.loads(args.model_args)
        for k, v in model_args.items():
            default_model_args[k] = v
        model_args = default_model_args
        self.model = LLM(
            model=args.model_path,
            **model_args
        )
        self.sampling_params = self._get_sampling_params(args)

    def _get_sampling_params(self, args):
        with open(os.path.join(os.path.dirname(__file__), "default_sampling_params.json"), "r") as f:
            default_config = json.load(f)
        try:
            config_path = os.path.join(os.path.dirname(__file__), "configs", args.task_name)
            with open(os.path.join(config_path, args.sampling_params_fn), "r") as f:
                sampling_kwargs = json.load(f)
        except FileNotFoundError:
            sampling_kwargs = {}
        for k, v in default_config.items():
            if k not in sampling_kwargs:
                sampling_kwargs[k] = v
        return SamplingParams(**sampling_kwargs)

    def generate(self, inputs: dict, use_tqdm=False):
        outputs = self.model.generate(inputs, self.sampling_params, use_tqdm=use_tqdm)
        generated_texts = [output.outputs[0].text.strip() for output in outputs]
        return generated_texts


class HfModel:
    def __init__(self, args) -> None:
        self.device = torch.device("cpu" if args.use_cpu else "cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.sampling_kwargs = self._get_sampling_params(args)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path).to(self.device)

    def _get_sampling_params(self, args):
        with open(os.path.join(os.path.dirname(__file__), "default_sampling_params.json"), "r") as f:
            default_config = json.load(f)
        try:
            config_path = os.path.join(os.path.dirname(__file__), "configs", args.task_name)
            with open(os.path.join(config_path, args.sampling_params_fn), "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        for k, v in default_config.items():
            if k not in config:
                config[k] = v
        if config.get("temperature", None) == 0:
            sampling_kwargs = {
                "max_new_tokens": config.get("max_tokens"),
                "do_sample": False
            }
        else:
            sampling_kwargs = {
                "temperature": config["temperature"],
                "max_new_tokens": config["max_tokens"],
                "top_p": config["top_p"],
                "top_k": config["top_k"],
                "do_sample": True,
            }
        return sampling_kwargs

    def generate(self, inputs: list[str], use_tqdm=False):
        generated_texts = []
        for input in tqdm(inputs, desc="infering", disable=not use_tqdm):
            prompt = input["prompt"]
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(
                **encoded_input,
                **self.sampling_kwargs
            )
            output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
            generated_texts.append(output[len(prompt):])
        return generated_texts


def initialize_model(args):
    model_type = {
        "hf": HfModel,
        "vllm": VllmModel
    }
    model = model_type[args.model_type](args)
    return model