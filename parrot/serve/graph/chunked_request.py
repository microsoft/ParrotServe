# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Union, Dict, Optional, List, Type
import re

from parrot.exceptions import parrot_assert, ParrotCoreUserError
from parrot.sampling_config import SamplingConfig


@dataclass
class RequestBodyChunk:
    """Base class of all chunks in the request body."""

    pos_id: int  # The position id of the chunk in the request. (0, 1, 2, ...)


@dataclass
class TextChunk(RequestBodyChunk):
    """A text chunk in the request body."""

    text: str  # The text of the chunk.


@dataclass
class PlaceholderNameChunk(RequestBodyChunk):
    """A placeholder in the request body."""

    name: str  # The name of the placeholder.


@dataclass
class RequestPlaceholder:
    """Detailed information of a placeholder in the request."""

    name: str
    is_output: bool
    var_id: Optional[str] = None
    sampling_config: Optional[Union[Dict, SamplingConfig]] = None

    def __post_init__(self) -> None:
        # Cast sampling_config to SamplingConfig.
        if self.sampling_config is not None:
            self.sampling_config = SamplingConfig(**self.sampling_config)

        # Check input/output arguments.
        if self.is_output:
            if self.var_id is not None:
                raise ValueError("Output placeholder should not have var_id.")

            # Default sampling_config for output placeholder.
            if self.sampling_config is None:
                self.sampling_config = SamplingConfig()
        else:
            if self.sampling_config is not None:
                raise ValueError("Input placeholder should not have sampling_config.")

    @property
    def has_var(self) -> bool:
        """Return whether the placeholder has an existing semantic variable."""

        return self.var_id is not None

    @property
    def should_create(self) -> bool:
        """Return whether we should created a new SV for this placeholder.

        Case 1: The placeholder is an output placeholder.
        Case 2: The placeholder is an input placeholder and has no value.
        """

        return self.is_output or not self.has_var


@dataclass
class RequestMetadata:
    """RequestMetadata contains metadata for a Request."""

    REQUEST_METADATA_KEYS = ["session_id", "models", "model_type", "remove_pure_fill"]

    session_id: int
    models: List[str]
    model_type: str
    remove_pure_fill: bool


class ChunkedRequest:
    """Parsed semantic call request.

    We firstly parse the prompt part into a list of text chunks and placeholders, and
    pack metadata and parsed prompt into a ChunkedRequest for further processing.
    """

    def __init__(self, metadata: RequestMetadata) -> None:
        # Metadata: additional information of the request.
        self.metadata = metadata

        # Body: the parsed prompt.
        self.body: List[RequestBodyChunk] = []

        # Placeholder map: map from placeholder name to placeholder.
        self.placeholders_map: Dict[str, RequestPlaceholder] = {}

    def push_chunk(self, chunk_type: Type[RequestBodyChunk], info: str) -> None:
        """Push a chunk into the body queue."""

        # Tricky here: both TextChunk and PlaceholderNameChunk are initialized using the same
        # function signature.

        pos_id = len(self.body)
        chunk = chunk_type(pos_id, info)
        self.body.append(chunk)

    def split_prefix_chunk(self, split_pos: int) -> None:
        """Split the prefix text chunk in specific position."""

        # Check the validity.
        prefix_chunk: TextChunk = self.body[0]
        parrot_assert(isinstance(prefix_chunk, TextChunk), "Invalid prefix chunk type.")

        # Split the prefix text chunk into two parts.
        prefix_text = prefix_chunk.text
        parrot_assert(0 <= split_pos < len(prefix_text), "Invalid split position.")

        prefix_chunk.text = prefix_text[:split_pos]
        new_prefix_chunk = TextChunk(1, prefix_text[split_pos:])
        self.body.insert(1, new_prefix_chunk)

        # Update the pos_id of the following chunks.
        for i in range(2, len(self.body)):
            self.body[i].pos_id += 1

    def __repr__(self) -> str:
        return (
            f"metadata: {self.metadata}, "
            f"body: {self.body}, "
            f"placeholders_map: {self.placeholders_map}"
        )

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @staticmethod
    def _preprocess(payload: Dict) -> Dict:
        """Preprocess the payload packet. This will do the format check and assign default values."""

        # Check format.
        parrot_assert("template" in payload, "Missing field 'template' in request.")
        parrot_assert(
            "placeholders" in payload, "Missing field 'placeholders' in request."
        )

        processed_payload = payload.copy()

        # Assign default values.
        processed_payload.setdefault("models", [])
        processed_payload.setdefault("model_type", "token_id")
        processed_payload.setdefault("remove_pure_fill", True)

        return processed_payload

    @classmethod
    def parse_from_payload(cls, payload: Dict) -> "ChunkedRequest":
        """Parse the payload of semantic call request into structural ChunkedRequest format for further processing.

        Args:
            payload: The payload of the HTTP packet.

        Returns:
            The parsed request.
        """

        payload = cls._preprocess(payload)

        # Pre-defined regex of placeholder name.
        PLACEHOLDER_REGEX = "{{[a-zA-Z_][a-zA-Z0-9_]*}}"

        # Get arguments from payload packet.
        template: str = payload["template"]
        placeholders: Dict = payload["placeholders"]

        # Step 1. Packing metadata.
        metadata_dict = {}
        for key in RequestMetadata.REQUEST_METADATA_KEYS:
            metadata_dict[key] = payload[key]
        metadata = RequestMetadata(**metadata_dict)
        chunked_request = cls(metadata)

        # Step 2. Extract the "placeholders" field and create placeholders dict.
        for placeholder in placeholders:
            # Format check included in initialization. (See RequestPlaceholder.__post_init__.)
            try:
                parsed_placeholder = RequestPlaceholder(**placeholder)
            except BaseException as e:
                raise ParrotCoreUserError(e)

            placeholder_name = parsed_placeholder.name

            # No duplicate placeholder name in "placeholders" field.
            parrot_assert(
                placeholder_name not in chunked_request.placeholders_map,
                "Duplicate placeholder name.",
            )
            chunked_request.placeholders_map[placeholder_name] = parsed_placeholder

        # Step 3. Parse prompt body.

        # Match all placeholders.
        pattern = re.compile(PLACEHOLDER_REGEX)

        iterator = pattern.finditer(template)
        last_pos = 0

        # For every matched placeholder: "abcd {YYY} efg", we first push "abcd" into body queue using
        # "last_pos: matched.start()", then push the placeholder into the body queue.
        # (Note that constant text "efg" will be pushed into body queue in the next iteration.)
        for matched in iterator:
            # Constant
            prev_text_chunk = template[last_pos : matched.start()]
            if (
                prev_text_chunk != ""
            ):  # Special case: if there is no constant text before the placeholder.
                chunked_request.push_chunk(TextChunk, prev_text_chunk)

            # Placeholder
            placeholder_name = template[matched.start() + 2 : matched.end() - 2]
            chunked_request.push_chunk(PlaceholderNameChunk, placeholder_name)

            # Update last_pos for the next iteration.
            last_pos = matched.end()

        # Push the last constant text chunk (after the last placeholder) into the body queue.
        last_text_chunk = template[last_pos:]
        if last_text_chunk != "":
            chunked_request.body.append(TextChunk(text=last_text_chunk))

        return chunked_request
