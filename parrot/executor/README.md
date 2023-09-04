# Executor in Parrot

The executor of the semantic program optimizer *Parrot* is responsible for 
- Manage sessions created by semantic funtion calls.
- Submit requests to backend server.
- Collect executed results from backend.

The main characteristics are:
- Variable-level asynchronization.
- Token-level pipelining message passing through tunnel.

##### Motivation

**Batching is important in LLM inference**. Existing LLM Agent SDK only supports
request-level asynchronization mainly because the constraint of API, resulting small batches 
and low GPU efficiency.

The executor aims to bridge between frontend and backend (LLM execution engine), providing sub-sequence asynchronization which is more friendly to LLM execution.