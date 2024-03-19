# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional, List

from parrot.serve.graph.nodes import ConstantFill, PlaceholderFill, PlaceholderGen
from parrot.serve.graph.completion_chain import GenTask
from parrot.exceptions import parrot_assert
from parrot.protocol.internal.primitive_request import Primitive, Fill, Generate

from ..tokenizer_wrapper import TokenizersWrapper


class RequestBuilder:
    """
    The request builder is responsible for transforming the graph nodes to real requests
    that can be executed by the backend, i.e. Fill/Generate primitives.
    """

    def __init__(
        self,
        tokenizer_name: Optional[str] = None,
        tokenizer: Optional[TokenizersWrapper] = None,
    ):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer

    def transform(self, gen_task: GenTask) -> List[Primitive]:
        """Transform the GenTask to Fill/Generate primitives.

        Args:
            gen_task: The GenTask to be transformed.
        """

        if self.tokenizer is not None:
            eos_token_id = self.tokenizer.get_tokenizer(
                self.tokenizer_name
            ).eos_token_id

        ret: List[Primitive] = []

        for i, node in enumerate(gen_task.fill_nodes):
            if isinstance(node, ConstantFill) or isinstance(node, PlaceholderFill):
                parrot_assert(
                    node.sv.ready,
                    "Placeholder should be ready when transformed to request.",
                )
                text = node.sv.get()

                if self.tokenizer is None:
                    fill = Fill(text=text)
                else:
                    token_ids = self.tokenizer.tokenize(text, self.tokenizer_name)
                    fill = Fill(token_ids=token_ids)
                ret.append(fill)

            elif isinstance(node, PlaceholderGen):
                sampling_config = node.sampling_config

                # If not ignore_tokenizer_eos, we should add eos_token_id to stop_token_ids
                if not sampling_config.ignore_tokenizer_eos:
                    sampling_config.stop_token_ids.append(eos_token_id)

                ret.append(Generate(sampling_config=sampling_config))

        return ret
