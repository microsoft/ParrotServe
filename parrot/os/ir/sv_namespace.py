# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import uuid


class SemanticVariableNamespace:
    """A namespace of Semantic Variables, giving a unique id to each SV."""

    def __init__(self):
        self._counter = 0  # This is not recycled. It just keeps increasing.
        self._namespace_uuid = uuid.uuid4() # A UUID object.
        self._ids = set()

    def get_new_id(self) -> str:
        """Get a new (unused) id for a SV."""

        self._counter += 1
        # return str(
        #     uuid.uuid3(
        #         namespace=self._namespace_uuid,
        #         name=str(self._counter),
        #     )
        # )
        return str(self._counter)  # For easier debugging