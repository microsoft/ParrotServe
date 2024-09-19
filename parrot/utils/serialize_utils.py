# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

import base64


def bytes_to_encoded_b64str(input_bytes: bytes) -> str:
    """Convert bytes to string using base64 ASCII encoding.
    
    Args:
        input_bytes (bytes): input bytes
    
    Returns:
        str: base64 string
    """

    b64str = str(base64.b64encode(input_bytes), encoding="ascii")  # base64 string
    return b64str


def encoded_b64str_to_bytes(b64str: str) -> bytes:
    """Convert base64 string (encoded by method "bytes_to_str") to bytes.
    """

    bytes_raw = bytes(b64str, encoding="ascii")
    b64bytes = base64.b64decode(bytes_raw)
    return b64bytes

