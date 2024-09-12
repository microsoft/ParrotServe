# Engine Management

In Parrot, the cluster manages various `Engine`s by class `EngineManager` in the `ServeCore`.

## Engine and Model

For Parrot `ServeCore`, `Engine` is just an abstraction of backend LLMs and their servers.
- After launching the `Engine` and specifying the address of the `ServeCore` in the config file of the engine, the engine will try to send a `register` request to the `ServeCore`.
- The `ServeCore` continuously receives heartbeats from each `Engine`. If it doesn't receive heartbeat from some `Engine`s for a configurable time interval, it will remove them from the manager.


In Parrot, each `Engine` runs a single `Model`, so the `EngineManager` is responsible for mapping the `Model` to the `Engine`.

To be specific, when a new `Engine` is registered to the manager, the manager will check whether the `Model` exists. If not, the manager will store the `Model` information and reuse it next time a new `Engine` with this model is registered. Maintaining such `Model`s list has many benefits. For example, we can expose current `Model`s to users or administrators, which show the supported `Model`s currently in our system, and this list is dynamically maintained when new `Engine`s come in and out.

## Exception Handling

When there are exceptions/errors raised during execution (in the `GraphExecutor`), we need to handle them. For now, we just use a simple strategy that we consider exceptions raised from the `Engine` side are all unrecoverable. So we report them to the upper layer and mark the corresponding `Engine` as "bad". The "bad" engines will be automatically removed in the `serve_loop`.
