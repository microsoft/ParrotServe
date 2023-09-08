from typing import Dict, List

from .controller import Controller
from ..program.function import ParrotFunction, Constant, Prefix


class TokenizedStorage:
    """Store the tokenized part of some parts of functions."""

    def __init__(self, controller: Controller):
        self.controller = controller
        # (Function Name, tokenizer) -> Function Body -> Token ids
        self.storage: Dict[(str, str), List[List[int]]] = {}

    def _tokenizer(self, tokenizer_name: str):
        return self.controller.tokenizers_table[tokenizer_name]

    def tokenize_func_body(
        self,
        function: ParrotFunction,
        tokenizer_name: str,
    ) -> List[int]:
        key = (function.name, tokenizer_name)
        if key not in self.storage:
            tokenizer = self._tokenizer(tokenizer_name)

            tokenized: List[List[int]] = []
            for piece in function.body:
                if isinstance(piece, Constant):
                    tokenized.append(
                        tokenizer.encode(
                            piece.text,
                            add_special_tokens=isinstance(piece, Prefix),
                        )
                    )
                else:
                    tokenized.append([])  # Empty for var loc
            self.storage[key] = tokenized

        return self.storage[key].copy()  # Avoid modification

    # NOTE(chaofan): Ignore special tokens because we chunk the inputs.

    def tokenize(self, text: str, tokenizer_name: str) -> List[int]:
        tokenizer = self._tokenizer(tokenizer_name)
        return tokenizer.encode(text, add_special_tokens=False)

    def detokenize(
        self,
        token_ids: List[int],
        tokenizer_name: str,
    ) -> str:
        tokenizer = self._tokenizer(tokenizer_name)
        return tokenizer.decode(token_ids, skip_special_tokens=True)
