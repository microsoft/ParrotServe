import json
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np


class ArxivLoader:
    """Load arxiv dataset."""

    def __init__(self, tokenizer_name: str):
        self.data_path = f"arxiv.json"
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

    def sample_articles_length_greater_than(
        self, token_length: int, sample_num: int = 1
    ):
        pool = []
        for i, data_dict in enumerate(self.data):
            if len(data_dict["tokenized_text"]) < token_length:
                continue
            pool.append(i)

        if len(pool) >= sample_num:
            return np.random.choice(pool, sample_num, replace=False)

        raise ValueError(
            f"No enough articles with length greater than {token_length} found."
        )

    def generate_articles_length_greater_than(self, token_length: int, sample_num: int):
        data_sorted = sorted(self.data, key=lambda x: len(x["tokenized_text"]))

        num_sampled = 0
        for _, data_dict in enumerate(data_sorted):
            if len(data_dict["tokenized_text"]) < token_length:
                continue
            yield data_dict
            num_sampled += 1
            if num_sampled >= sample_num:
                break


if __name__ == "__main__":
    loader = ArxivLoader(tokenizer_name="hf-internal-testing/llama-tokenizer")
    loader.load()

    indices = loader.sample_articles_length_greater_than(20000, 20)

    # for i, index in enumerate(indices):
    #     with open(f"arxiv-sampled-1/article_{i}.txt", encoding="utf-8", mode="w") as f:
    #         print(len(loader.data[index]["tokenized_text"]))
    #         f.write(loader.data[index]["text"])

    counter = 0
    for data_dict in loader.generate_articles_length_greater_than(20000, 30):
        with open(
            f"arxiv-sampled-1/article_{counter}.txt", encoding="utf-8", mode="w"
        ) as f:
            print(len(data_dict["tokenized_text"]))
            f.write(data_dict["text"])
            counter += 1
