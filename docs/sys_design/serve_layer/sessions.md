# Session Management

`Session` represents a user’s connection to Parrot. It is exposed to the Parrot frontend API and is manually created and deleted by the user.

Overall, managing `Session`s is quite similar to managing the `Engine`s, with the following key similarities:
- Both of them are created explicitly.
- Both of them have an expiration mechanism.

There is still a little difference between `Session`'s expiration mechanism and `Engine`s. The latter one is based on the heartbeat, while `Session` updates their last seen time by each API call in this session. Everytime the `ServeCore` receives a request with its `session_id`, the core resets the session's expiration time.

A `Session` has its own `ComputeGraph`, `GraphExecutor`. And when the `Session` is freed, the Semantic Variables and Contexts belonging to the session's namespace are accordingly freed. For the concept of "namespace", see [Semantic Variable Management](sv_manage.md).