# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List

from parrot.exceptions import parrot_assert
from parrot.protocol.internal.primitive_request import Primitive, Fill, Generate


class PrimitiveBuilder:
    """
    The request builder is responsible for transforming the graph nodes to real requests
    that can be executed by the backend, i.e. Fill/Generate primitives.
    """

    def __init__(self, session_id: int) -> None:
        self.session_id = session_id

    def transform(self, completion_task: CompletionTask) -> List[Primitive]:
        """Transform the GenTask to Fill/Generate primitives.

        Returns:
            List[Primitive]: The list of primitives to be executed.
        """

        parrot_assert(
            completion_task.engine is not None, "Execution engine is not available."
        )

        requires_token_ids = completion_task.engine.model_type
        if requires_token_ids:
            parrot_assert(
                completion_task.is_tokenized, "Tokenized result is not available."
            )
            tokenizer_name = completion_task.engine.requires_token_ids

        ret: List[Primitive] = []

        for task_node in completion_task.task_body:
            if task_node.node.is_gen:
                primitive = Generate(
                    session_id=self.session_id,
                    task_id=completion_task.task_id,
                    context_id=task_node.context.context_id,
                    parent_context_id=task_node.context.parent_context_id,
                    end_flag=False,
                    sampling_config=task_node.node.sampling_config,
                )
            else:
                if requires_token_ids:
                    token_ids = completion_task.tokenized_result[tokenizer_name][
                        task_node.idx
                    ]
                    primitive = Fill(
                        session_id=self.session_id,
                        task_id=completion_task.task_id,
                        context_id=task_node.context.context_id,
                        parent_context_id=task_node.context.parent_context_id,
                        end_flag=False,
                        token_ids=token_ids,
                    )
                else:
                    primitive = Fill(
                        session_id=self.session_id,
                        task_id=completion_task.task_id,
                        context_id=task_node.context.context_id,
                        parent_context_id=task_node.context.parent_context_id,
                        end_flag=False,
                        text=task_node.node.get(),
                    )

            ret.append(primitive)

        return ret
