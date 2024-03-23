# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Dict, Optional

from parrot.serve.backend_repr import ExecutionEngine
from parrot.serve.graph import CompletionChain


class ScheduleUnit:
    """ScheduleUnit wraps CompletionChain."""

    def __init__(self, task_id: int, chain: CompletionChain):
        self.task_id = task_id
        self.chain = chain

        # Tokenized result
        # Map from tokenizer name to tokenized result
        # A tokenized result is a List of token ids, i.e. List[List[int]]
        self.tokenized_result: Optional[Dict[str, List[List[int]]]] = None

        # Scheduled result
        self.engine: Optional[ExecutionEngine] = None

    @property
    def is_tokenized(self) -> bool:
        return self.tokenized_result is not None

    def tokenize_chain(self, tokenizers_wrapper: "TokenizersWrapper"):
        """Tokenize the chain using the tokenizers in the wrapper."""

        self.tokenized_result = {}
        for fill_node in self.chain.iter_fill():
            tokenized_result: Dict = tokenizers_wrapper.tokenize_all(fill_node.get())
            for key, value in tokenized_result.items():
                if key not in self.tokenized_result:
                    self.tokenized_result[key] = []
                self.tokenized_result[key].append(value)

    def get_token_nums(self, tokenizer_name: str) -> int:
        """Get the number of tokens in the tokenized result."""

        parrot_assert(self.is_tokenized, "Tokenized result is not available.")
        tokens_num = 0
        for token_ids in self.tokenized_result[tokenizer_name]:
            tokens_num += len(token_ids)
        return tokens_num

    def __str__(self):
        return f"CompletionTask(chain={self.chain})"


class GlobalScheduler:
    """GlobalScheduler(GS) solves the task scheduling problem in the global scope.

    A scheduling unit is a
    """

    def __init__(self):
        pass
