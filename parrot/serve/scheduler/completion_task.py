# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List, Optional

from ..graph.node_struct import CompletionChain


class CompletionTask:
    """CompletionTask represents a task to be completed by the scheduler."""

    def __init__(self, task_id: int, chain: CompletionChain):
        self.task_id = task_id
        self.chain = chain

        # Tokenized result
        # Map from tokenizer name to tokenized result
        # A tokenized result is a List of token ids, i.e. List[List[int]]
        self.tokenized_result: Optional[Dict[str, List[List[int]]]] = None

    @property
    def is_tokenized(self) -> bool:
        return self.tokenized_result is not None

    def tokenize_chain(self, tokenizers_wrapper: "TokenizersWrapper"):
        """Tokenize the chain using the tokenizers in the wrapper."""

        self.tokenized_result = {}
        for fill_node in self.chain.iter_fill():
            tokenized_result = tokenizers_wrapper.tokenize_all(fill_node.get())
            for key, value in tokenized_result.items():
                if key not in self.tokenized_result:
                    self.tokenized_result[key] = []
                self.tokenized_result[key].append(value)
    
    def count_

    def __str__(self):
        return f"CompletionTask(chain={self.chain})"
