# Reference:
#     https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py

from typing import List
from enum import auto, IntEnum

from .func_mutator import (
    FuncMutator,
    SemanticFunction,
    Constant,
    Parameter,
)


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
