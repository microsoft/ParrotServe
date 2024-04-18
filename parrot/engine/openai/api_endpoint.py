# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum, auto


class Endpoint(Enum):
    """To distinguish between different OpenAI endpoints.

    For text generation, we have two endpoints: `completion` and `chat`.

    Check: https://platform.openai.com/docs/api-reference/ for more details.
    """

    COMPLETION: int = auto()
    CHAT: int = auto()


ENDPOINT_MAP = {
    "completion": Endpoint.COMPLETION,
    "chat": Endpoint.CHAT,
}
