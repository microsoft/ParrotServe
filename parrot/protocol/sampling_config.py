# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class SamplingConfig:
    """SamplingConfig is a set of parameters for LLM sampling."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_gen_length: int = 512
    ignore_tokenizer_eos: bool = False
    stop_token_ids: List[int] = field(default_factory=list)
    stop_str: Optional[str] = None

    # The following configs are only used in OpenAI engine for now.
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    n: int = 1
    best_of: int = 1
    logit_bias: Optional[Dict[str, int]] = None

    # The following configs are not used for now.
    repetition_penalty: float = 0.0
    length_penalty: float = 0.0

    def get_openai_params(self) -> Dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_gen_length,
            "stop": self.stop_str,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            # "n": self.n,
            # "best_of": self.best_of,
            # "logit_bias": self.logit_bias,
        }
