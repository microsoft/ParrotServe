# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

class PrefixCache:
    """Prefix cache maps a prefix hash to a semantic variable.

    A prefix hash consists of constant part and variable part. The constant part is directly 
    hashed according to text content, and the variable part is hashed according to the semantic 
    variable id.
    """

    