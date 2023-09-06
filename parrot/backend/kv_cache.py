from typing import List
from dataclasses import dataclass


@dataclass
class KVContext:
    """Low-level implementation of Context."""

    context_id: int
    block_ids: List[int]
    last_token_id: int
    parent_context: "KVContext"
