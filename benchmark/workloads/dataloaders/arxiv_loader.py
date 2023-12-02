import json
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np


import parrot


class ArxivLoader:
    """Load arxiv dataset."""

    def __init__(self, tokenizer_name: str):
        self.parrot_path = parrot.__path__[0]
        self.data_path = (
            f"{self.parrot_path}/../benchmark/workloads/arxiv-march-2023/arxiv.json"
        )
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def load(self):
        with open(self.data_path, encoding="utf-8", mode="r") as f:
            lines = f.readlines()

        article_str_lengths = []
        article_token_lengths = []

        for line in tqdm(lines):
            data_dict = json.loads(line)
            tokenized_text = self.tokenizer.encode(data_dict["text"])
            data_dict["tokenized_text"] = tokenized_text
            self.data.append(data_dict)

            article_str_lengths.append(len(data_dict["text"]))
            article_token_lengths.append(len(tokenized_text))

        print(f"Average article length (str): {np.mean(article_str_lengths):.2f}")
        print(f"Average article length (token): {np.mean(article_token_lengths):.2f}")
        print(f"Max article length (str): {np.max(article_str_lengths)}")
        print(f"Max article length (token): {np.max(article_token_lengths)}")
        print(f"Min article length (str): {np.min(article_str_lengths)}")
        print(f"Min article length (token): {np.min(article_token_lengths)}")

    def get_sample_with_length_greater_than(self, token_length: int):
        for data_dict in self.data:
            if len(data_dict["tokenized_text"]) > token_length:
                return data_dict


if __name__ == "__main__":
    loader = ArxivLoader(tokenizer_name="hf-internal-testing/llama-tokenizer")
    loader.load()

    lengths = []
