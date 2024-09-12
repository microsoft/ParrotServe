# Graph Executor

Parrot adopts a graph executor to enable automatically parallelizing and batching LLM requests in the DAG. Each `Session` has its own executor, with a `ComputeGraph` dynamically maintained.

## Graph-based Execution with Coroutines

Parrot's `ComputeGraph` is a data-dependent graph. The basic unit of execution is `CompletionChain` in our graph (See [Graph](graph.md)), which will be executed once it's ready ("Ready" means the dependencies of the chain have all been executed).

To implement this, we need to continuously poll and pop out chains with zero in-degree. Parrot assigns a `Coroutine` to each `CompletionChain`, and wraps it as a task in the polling loop. Different CompletionChains communicate with each other using `Event`s (in Python asynchronous programming framework).