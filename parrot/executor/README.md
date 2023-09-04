# Executor in Parrot

The executor of the semantic program optimizer *Parrot* is responsible for 
- Manage sessions created by semantic funtion calls.
- Submit requests to backend server.
- Collect executed results from backend.

The main characteristics are:
- Variable-level asynchronization.
- Token-level pipelining message passing through tunnel.

### Motivation

**Asynchronous is important in semantic programming.** Similar to network programming:
- We send requests to certain servers to get resources we need.
- Waiting for requests may take a long time. (Latency)
- The result is not reliable. (404, Retry)

Asynchronous programming is popular in network programming (JavaScript, Node, Python) because 
it lets the procedure of waiting for requests to be non-blocking. We point out that this is 
also very important in semantic programming.

**Batching is important in LLM inference**. Existing LLM Agent SDK only supports
request-level asynchronization mainly because the constraint of API, resulting small batches 
and low GPU efficiency. Sub-sequence asynchronization is more friendly to LLM execution, bringing 
us a larger batch in local serving scenario.