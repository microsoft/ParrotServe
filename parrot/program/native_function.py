# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Callable


class NativeFunction:
    def __init__(self, name, native_function):
        self.name = name
        self.native_function = native_function
