# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Union, Dict, Optional, List, Type
import re

from parrot.exceptions import parrot_assert, ParrotCoreUserError
from parrot.sampling_config import SamplingConfig

from .perf_criteria import PerformanceCriteria


# ------------------------ Semantic Call Request ------------------------


@dataclass
class SemanticCallRequestBodyChunk:
    """Base class of all chunks in the request body."""

    pos_id: int  # The position id of the chunk in the request. (0, 1, 2, ...)


@dataclass
class TextChunk(SemanticCallRequestBodyChunk):
    """A text chunk in the request body."""

    text: str  # The text of the chunk.


@dataclass
class PlaceholderChunk(SemanticCallRequestBodyChunk):
    """A placeholder in the request body."""

    name: str  # The name of the corresponding parameter.


@dataclass
class SemanticFunctionParameter:
    """Detailed information of a parameter in the function of the semantic call request.
    
    In semantic function, a parameter can also be considered as a "placeholder" in the prompt.
    """

    name: str
    is_output: bool
    var_id: Optional[str] = None
    sampling_config: Optional[Union[Dict, SamplingConfig]] = None

    def __post_init__(self) -> None:
        # Cast sampling_config to SamplingConfig.
        if isinstance(self.sampling_config, dict):
            self.sampling_config = SamplingConfig(**self.sampling_config)

        # Check input/output arguments.
        if self.is_output:
            if self.var_id is not None:
                raise ValueError("Output parameter should not have var_id.")

            # Default sampling_config for output parameter.
            if self.sampling_config is None:
                self.sampling_config = SamplingConfig()
        else:
            if self.sampling_config is not None:
                raise ValueError("Input parameter should not have sampling_config.")

    @property
    def has_var(self) -> bool:
        """Return whether the parameter has an existing semantic variable."""

        return self.var_id is not None

    @property
    def should_create(self) -> bool:
        """Return whether we should created a new SV for this parameter.

        Case 1: The parameter is an output parameter.
        Case 2: The parameter is an input parameter and has no value.
        """

        return self.is_output or not self.has_var


@dataclass
class SemanticCallMetadata:
    """SemanticCallMetadata contains metadata for a Request."""

    REQUEST_METADATA_KEYS = [
        "models",
        "model_type",
        "remove_pure_fill",
        "cache_prefix",
        "output_criteria",
        "fuse_fill",
    ]

    models: List[str]
    model_type: str
    remove_pure_fill: bool
    cache_prefix: bool
    output_criteria: Optional[Union[PerformanceCriteria, str]]
    fuse_fill: bool

    @classmethod
    def get_default_dict(cls) -> Dict:
        """Get the default metadata for a Request in dict format."""

        return {
            "models": [],
            "model_type": "token_id",
            "remove_pure_fill": True,
            "cache_prefix": True,
            "output_criteria": None,
            "fuse_fill": False,
        }

    @classmethod
    def get_default(cls) -> "SemanticCallMetadata":
        """Get the default metadata for a Request."""

        return SemanticCallMetadata(**cls.get_default_dict())


class ChunkedSemanticCallRequest:
    """Parsed semantic call request.

    We firstly parse the prompt part into a list of text chunks and placeholders, and
    pack metadata and parsed prompt into a ChunkedSemanticCallRequest for further processing.
    """

    def __init__(
        self,
        request_id: int,
        session_id: int,
        metadata: SemanticCallMetadata = SemanticCallMetadata.get_default(),
    ) -> None:
        self.request_id = request_id
        self.session_id = session_id

        # Metadata: additional information of the request.
        self.metadata = metadata

        # Body: the parsed prompt.
        self.body: List[SemanticCallRequestBodyChunk] = []

        # Parameters map: map from parameter name to parameter.
        self.parameters_map: Dict[str, SemanticFunctionParameter] = {}

    def push_chunk(self, chunk_type: Type[SemanticCallRequestBodyChunk], info: str) -> None:
        """Push a chunk into the body queue."""

        # Tricky here: both TextChunk and parameterNameChunk are initialized using the same
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
            f"parameters_map: {self.parameters_map}"
        )

    @staticmethod
    def _preprocess(payload: Dict) -> Dict:
        """Preprocess the payload packet. This will do the format check and assign default values."""

        # Check format.
        parrot_assert("template" in payload, "Missing field 'template' in request.")
        parrot_assert(
            "parameters" in payload, "Missing field 'parameters' in request."
        )

        processed_payload = payload.copy()

        # Assign default values.
        processed_payload.setdefault("models", [])
        processed_payload.setdefault("model_type", "token_id")
        processed_payload.setdefault("remove_pure_fill", True)

        return processed_payload

    @classmethod
    def parse_from_payload(
        cls, request_id: int, session_id: int, payload: Dict
    ) -> "ChunkedSemanticCallRequest":
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
        parameters: Dict = payload["parameters"]

        # Step 1. Packing metadata.
        metadata_dict = SemanticCallMetadata.get_default_dict()
        for key in SemanticCallMetadata.REQUEST_METADATA_KEYS:
            if key in payload:
                metadata_dict[key] = payload[key]
        metadata = SemanticCallMetadata(**metadata_dict)

        chunked_request = cls(request_id, session_id, metadata)

        # Step 2. Extract the "parameters" field and create parameters dict.
        for parameter in parameters:
            # Format check included in initialization. (See FunctionParameter.__post_init__.)
            try:
                parsed_parameter = SemanticFunctionParameter(**parameter)
            except BaseException as e:
                raise ParrotCoreUserError(e)

            parameter_name = parsed_parameter.name

            # No duplicate parameter name in "parameters" field.
            parrot_assert(
                parameter_name not in chunked_request.parameters_map,
                "Duplicate parameter name.",
            )
            chunked_request.parameters_map[parameter_name] = parsed_parameter

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
            parameter_name = template[matched.start() + 2 : matched.end() - 2]
            chunked_request.push_chunk(PlaceholderChunk, parameter_name)

            # Update last_pos for the next iteration.
            last_pos = matched.end()

        # Push the last constant text chunk (after the last placeholder) into the body queue.
        last_text_chunk = template[last_pos:]
        if not metadata.remove_pure_fill and last_text_chunk != "":
            chunked_request.push_chunk(TextChunk, last_text_chunk)

        return chunked_request


# ------------------------ Native Call Request ------------------------


@dataclass
class NativeCallMetadata:
    """Metadata of a native function."""

    timeout: (
        float  # If the function execution surpass this timeout, it will be terminated.
    )

    @classmethod
    def get_default_dict(cls) -> Dict:
        """Get the default metadata for a Request in dict format."""

        return {
            "timeout": 999999,
        }

    @classmethod
    def get_default(cls) -> "SemanticCallMetadata":
        """Get the default metadata for a Request."""

        return NativeCallMetadata(**cls.get_default_dict())


class PyNativeCallRequest:
    """Python native call request."""

    def __init__(
        self,
        request_id: int,
        session_id: int,
        metadata: NativeCallMetadata = NativeCallMetadata.get_default(),
    ) -> None:
        self.request_id = request_id
        self.session_id = session_id

        # Metadata: additional information of the request.
        self.metadata = metadata

        # Body: the parsed prompt.

        # Parameters map: map from parameter name to parameter.
        self.parameters_map: Dict[str, SemanticFunctionParameter] = {}