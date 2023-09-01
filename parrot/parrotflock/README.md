# Parrotflock

Parrotflock (a flock of parrots) implements the middle part of the semantic program 
optimizer *Parrot*, including:
- High-efficient communications between semantic functions.
- Building DAG between parrot operators.

The main characteristics are:
- Variable-level asynchronization.
- Token-level pipelining message passing through tunnel.

Motivation: **Batching is important in LLM inference**. Existing LLM Agent SDK only supports
request-level asynchronization mainly because the constraint of API, resulting small batches 
and low GPU efficiency.

Parrotflock aims to bridge between frontend and backend (LLM execution engine), providing sub-sequence asynchronization which is more friendly to LLM execution.