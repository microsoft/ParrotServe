# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

import base64
import marshal

from types import CodeType


def bytes_to_encoded_b64str(serialized: bytes) -> str:
    """Convert bytes to string using base64 ASCII encoding.

    Args:
        input_bytes (bytes): input bytes

    Returns:
        str: base64 string
    """

    # Encode the serialized code to base64
    encoded_code = base64.b64encode(serialized).decode("utf-8")
    return encoded_code


def encoded_b64str_to_bytes(b64str: str) -> bytes:
    """Convert base64 string (encoded by method "bytes_to_str") to bytes."""

    b64bytes = base64.b64decode(b64str)
    return b64bytes


def serialize_func_code(pyfunc_code: CodeType) -> bytes:
    """Serialize a Python function code to bytes."""

    # Serialize the function
    serialized_code = marshal.dumps(pyfunc_code)
    return serialized_code


def deserialize_func_code(serialized_code: bytes) -> CodeType:
    """Deserialize a Python function code from bytes."""

    # Deserialize the function
    code = marshal.loads(serialized_code)
    return code
