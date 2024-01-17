# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List, Union
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


HFTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class Tokenizer:
    """Store the tokenized part of some parts of functions."""

    def __init__(self):
        self.tokenizers: Dict[str, HFTokenizer] = {}

    def get_tokenizer(self, tokenizer_name: str):
        if tokenizer_name not in self.tokenizers:
            self.tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(
                tokenizer_name
            )
        return self.tokenizers[tokenizer_name]

    # NOTE(chaofan): Ignore special tokens because we chunk the inputs.

    def tokenize(self, text: str, tokenizer_name: str) -> List[int]:
        tokenizer = self.get_tokenizer(tokenizer_name)
        return tokenizer.encode(text, add_special_tokens=False)

    def detokenize(
        self,
        token_ids: List[int],
        tokenizer_name: str,
    ) -> str:
        tokenizer = self.get_tokenizer(tokenizer_name)
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
