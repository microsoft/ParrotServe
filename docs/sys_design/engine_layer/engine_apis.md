# Internal APIs between ServeCore and Engine

Code: `parrot/protocol/internal/`.


## Control APIs

### Serve Layer to Engine Layer

- `/free_context`, arguments: `context_id: int`. Free a Low-level context in the engine.
- `/ping_engine`, arguments: None. Ping an engine to make sure it's alive.

### Engine Layer to Serve Layer

- `/register_engine`, arguments: `engine_config: EngineConfig`. Register an Engine in the ServeCore.
- `/engine_heartbeat`, arguments: `engine_id: int, engine_name: str, runtime_info: EngineRuntimeInfo`. Heartbeats from Engine, also update the runtime information of the engine.

## Primitive Requests

Primitive requests are APIs we have defined for implementing the basic functions of LLMs, primarily to support Contextual Prefill/Generate functionalities.

```python
class Primitve:
    # The session id.
    session_id: int
    
    # Its task id. Since two primitive requests belonging to the same CompletionTask cannot appear simultaneously in the Engine.
    task_id: int

    # Specify the Context this primitive operates on
    context_id: int
    parent_context_id: int
```

- `Fill` object (post on `/fill`):
    ```python
    class Fill(Primitive):
        token_ids: Optional[List[int]]
        text: Optional[str]
    ```
    A `Fill` can use a untokenized text `string` or a list of tokenized `token_ids` as the fill content, depending on the backend type the user choose. We don't call it `Prefill` since we support contextual `Fill` here, i.e., we can perform a `Fill` even after a `Generate`.

    When the ServeCore send a `Fill` request to an Engine, the Engine will calculate the KV cache on the specified `Context` (which can be viewed as we "extend" the `Context` by some tokens).

- `Generate` object (post on `/generate`).
    ```python
    class Generate(Primitive):
        sampling_configs: SamplingConfig
    ```
    This request will trigger a completion action based on the `specified` Context on the target Engine. The `Context` is also "extended" As tokens are generated one by one and the KV are appended to the corresponding KV cache.
    - `/generate_stream` (TODO)

Note: In fact, `free_context` can also be considered a type of primitive request, as it provides basic functionality for managing the context.