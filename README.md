# Avesync

Avesync is an asynchronous task scheduler for semantic programming, which implements:
- Variable-level asynchronization.
- Token-level asynchronization.

Motivation: **Batching is important in LLM inference**. Existing LLM Agent SDK only supports
request-level asynchronization mainly because the constraint of API, resulting small batching 
and low GPU efficiency due to control-flow blocking.

Avesync aims to bridge between frontend and backend (LLM execution engine), providing sub-sequence 
asynchronization which is more friendly to LLM execution.