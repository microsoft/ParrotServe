from typing import List
from dataclasses import dataclass


@dataclass
class InputMetadata:
    seq_ids: List[int]

    """Structure of a sequence:
    
    |<--- recompute --->|<--- prefill --->|

    or

    |<--- recompute --->|<--- cached --->|<--- prefill --->|
    (this case, the prefill part is the next-token)
    
    - recompute: compute -> compute
    - prefill: compute -> cached
    - cached: cached -> cached
    """

    recompute_lens: List[int]
    prefill_lens: List[int]
    cached_lens: List[int]
    lens: List[int]


@dataclass
class Sequence:
    seq_id: int
    token_ids: List[int]
    prefilled: bool = False

    @property
    def seq_len(self) -> int:
        return len(self.token_ids)
