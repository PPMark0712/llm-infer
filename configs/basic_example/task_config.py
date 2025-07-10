import os

def get_file_list(input_path):
    return [os.path.join(os.path.dirname(__file__), "test_data.txt")]


def read_file(file_path):
    with open(file_path, "r") as f:
        return [{"text": line.strip()} for line in f.readlines()]


def format_prompt(org_data):
    return org_data["text"]


def extract_answer(model_response):
    return None