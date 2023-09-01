# Parrotalk

Parrotalk implements the middle part of the semantic program optimizer *Parrot*, which 
provides high-efficient **communications between semantic functions**.

It implements:
- Variable-level asynchronization.
- Token-level pipelining message passing through tunnel.

Motivation: **Batching is important in LLM inference**. Existing LLM Agent SDK only supports
request-level asynchronization mainly because the constraint of API, resulting small batching 
and low GPU efficiency due to control-flow blocking.

Parrotalk aims to bridge between frontend and backend (LLM execution engine), providing sub-sequence 
asynchronization which is more friendly to LLM execution.