from typing import Dict, List

from .controller import Controller, parrot_global_ctrl
from ..program.function import ParrotFunction, Constant


class TokenizedStorage:
    """Store the tokenized part of some parts of functions."""

    def __init__(self):
        # (Function Name, tokenizer) -> Function Body -> Token ids
        self.storage: Dict[(str, str), List[List[int]]] = {}

    def tokenize_func_body(
        self,
        function: ParrotFunction,
        tokenizer_name: str,
    ) -> List[int]:
        key = (function.name, tokenizer_name)
        if key not in self.storage:
            tokenizer = parrot_global_ctrl.tokenizers_table[tokenizer_name]

            tokenized: List[List[int]] = []
            for piece in function.body:
                if isinstance(piece, Constant):
                    tokenized.append(tokenizer(piece.text)["input_ids"])
                else:
                    tokenized.append([])  # Empty for var loc
            self.storage[key] = tokenized

        return self.storage[key].copy()  # Avoid modification


def detokenize(
    token_ids: List[int],
    tokenizer_name: str,
) -> str:
    tokenizer = parrot_global_ctrl.tokenizers_table[tokenizer_name]
    return tokenizer.decode(token_ids)


# Singleton
global_tokenized_storage = TokenizedStorage()
