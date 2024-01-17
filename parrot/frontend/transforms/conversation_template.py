# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List
from enum import auto, IntEnum
from dataclasses import asdict

from .func_mutator import (
    FuncMutator,
    SemanticFunction,
    Constant,
    SemanticRegion,
    ParameterLoc,
    Parameter,
)
from ..function import push_to_body


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()


class ConversationTemplate(FuncMutator):
    r"""Conversation template for open-source chat models.

    This transformation will transform normal functions into a conversation.
    The transform rule is:
        - Insert a system message into the beginning of the function.
        - For continuous Fill, transform them into a User message.
        - For a Generation loc, transform it into a Assistant message.
    """

    def __init__(
        self,
        system_message: str,
        roles: List[str],
        system_template: str = "{system_message}",
        seperator_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE,
        sep: str = "\n",
        sep2: str = "\n",
    ) -> None:
        # The content of the system prompt
        self.system_message = system_message
        # The roles of the conversations.
        # Must be a list of two strings. E.g. ["USER", "ASSISTANT"]
        self.roles = roles
        # The str template for the system prompt
        self.system_template = system_template
        # The separator style
        self.seperator_style = seperator_style
        self.sep = sep
        self.sep2 = sep2

    def _visit_constant(self, constant: Constant) -> Constant:
        return constant

    def _visit_parameter(self, param: Parameter) -> Parameter:
        return param

    def _visit_func(self, func: SemanticFunction) -> SemanticFunction:
        new_body: List[SemanticRegion] = []

        # Add system message
        push_to_body(
            Constant,
            new_body,
            text=self.system_template.format(system_message=self.system_message)
            + self.sep,
        )

        conversation_round_start_flag = True
        for piece in func.body:
            if conversation_round_start_flag:
                # Add user message
                push_to_body(
                    Constant,
                    new_body,
                    text=f"{self.roles[0]}: ",
                )
                conversation_round_start_flag = False

            is_output_loc = isinstance(piece, ParameterLoc) and piece.param.is_output
            if is_output_loc:
                # Add assistant message
                push_to_body(
                    Constant,
                    new_body,
                    text=f"{self.sep}{self.roles[1]}: ",
                )
                conversation_round_start_flag = True

            keys = list(piece.__dataclass_fields__.keys())
            keys.remove("idx")  # It will be set automatically
            data_dict = {k: getattr(piece, k) for k in keys}
            push_to_body(
                piece.__class__,
                new_body,
                **data_dict,
            )

            if is_output_loc:
                # Add assistant sep
                if self.seperator_style == SeparatorStyle.ADD_COLON_SINGLE:
                    sep = self.sep
                elif self.seperator_style == SeparatorStyle.ADD_COLON_TWO:
                    sep = self.sep2
                else:
                    raise ValueError(f"Unknown seperator style: {self.seperator_style}")

                push_to_body(
                    Constant,
                    new_body,
                    text=f"{sep}",
                )

        return SemanticFunction(
            name=func.name,
            params=func.params,
            func_body=new_body,
            **asdict(func.metadata),
        )


# The Vicuna chat template is from:
#   https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
vicuna_template = ConversationTemplate(
    system_message="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=["USER", "ASSISTANT"],
    seperator_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)
