# ParrotServeCore

`ServeCore` is the centralized manager of Parrot & **entry of the serve layer**, which manages Sessions (client-side), Engines (cluster-level), some related resources (virtual KV cache blocks, Semantic Variable, ...)

# Serve Loop

When Parrot is serving, `ServeCore` will run `serve_loop()` infinitely, at a fixed interval. On every iteration, `ServeCore` will do the following things:
- Check whether resources are expired (Engine, Session, Semantic Variable, ...), sweep dead items
- Try to schedule tasks in the queue of `GlobalScheduler`