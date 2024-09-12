# Context Management

## Context Object

`Context` is the high-level representation of memory (Specifically, KV cache). It can be either paged (paged-attention) or unpaged (past_key_values). We can fork a sub-context for a parent context, which provides API-support for prompt sharing.

**Definition.**

```python
class Context:
    context_id: int
    engine: Engine
    parent_context: Context
```

A `Context` object is a chunk of KV cache in a specific engine. Each `Context` object has a unique id and optionally a parent context (fork).

**Note: We don't store real KV cache in serve layer. The `Context` here is just for high-level management.**


## Prefix Cache

Prefix Cache maps a prefix hash to a context id.

THe method `_hash_var_id` of `ServeCoreContextManager` defines how to hash a single `SemanticVariable`. The key of the prefix cache map is just a long string with these hash strings concatenated. We only cache the prefix (a.k.a Consecutive `Fill`s at the beginning).

```
List of Semantic Variables -> Prefix Hash -> context id
```

## Allocate/Fork/Free Context

### Allocate

Implemented in `_new_context` method.

We allocate a `Context` for each Node of the chain (For Node/Chain, please refer to [the Graph part](graph.md)) so that different requests can dynamically and flexibly share and fork these `Context`s.

If a prefix hits the cache, we directly fork the `Context` in the cache.

### Fork

Implemented in `_fork_context` method.

`Fork` means we let different `Context`s share the same parent `Context`. We maintain a `ref_counter` in the manager, and for each fork we will let the `ref_counter` of the parent `Context` add by 1.

### Free

Implemented in `_free_context` method.

We will first decrease the `ref_counter` of this `Context` by 1. If the `ref_counter` is reduced to 0, we remove it from all maps in the manager/

## Context Info

We maintain `token_nums` of this `Context` in serve layer, by tracking the `Fill` and `Generate` execution (For `Generate`, we use `max_gen_length` as the tokens number).

## Global/Local Context

There are two kinds of contexts, distinguish by its scope: global context (constant prefix context) and local context (session context). The former one is assigned to the Constant Prefix Semantic Variable and the latter one is assigned to the Variables in each session's namespace.

For the Constant Prefix and different Variables' namespaces, see [Semantic Variable Management](sv_manage.md).