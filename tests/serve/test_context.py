import json

from parrot.serve.backend_repr import Context, ExecutionEngine, LanguageModel
from parrot.engine.config import EngineConfig
from parrot.testing.get_configs import get_sample_engine_config_path

from parrot.serve.variable_manager import SemanticVariableManager
from parrot.serve.scheduler import CompletionTask
from parrot.serve.context_manager import PrefixCache, ServeCoreContextManager
from parrot.sampling_config import SamplingConfig
from parrot.serve.graph import (
    RequestChain,
    ConstantFill,
    PlaceholderGen,
    PlaceholderFill,
)
from parrot.serve.graph.request import SemanticCallMetadata, RequestPlaceholder


def test_prefix_cache():
    svs = ["sv0", "sv1", "sv2"]
    prefix_cache = PrefixCache()
    prefix_hash = ""
    for context_id, sv in enumerate(svs):
        prefix_hash += ServeCoreContextManager._hash_var_id(sv)
        prefix_cache.cache_prefix_context(prefix_hash, context_id)
    print(prefix_cache._prefix_ctx_map)


def test_context_manager():
    session_id = 0
    var_mgr = SemanticVariableManager(666)
    var_mgr.register_local_var_space(session_id=0)
    var0 = var_mgr.create_var(session_id, "a")
    var0.set("Content0")

    request_chain = RequestChain.from_nodes(
        nodes=[
            ConstantFill("Test1"),
            PlaceholderFill(
                placeholder=RequestPlaceholder(
                    name="a", var_id=var0.id, is_output=False
                )
            ),
            ConstantFill("Test2"),
            PlaceholderGen(
                placeholder=RequestPlaceholder(
                    name="b", is_output=True, sampling_config=SamplingConfig()
                )
            ),
        ]
    )
    # request_chain.metadata.fuse_fill = True
    var_mgr.create_vars_for_request(session_id, request_chain)

    task = CompletionTask(task_id=0, chain=request_chain.comp_chains[0])

    config_path = get_sample_engine_config_path("opt-13b.json")
    with open(config_path, "r") as f:
        engine_config = EngineConfig.from_dict(json.load(f))
    engine = ExecutionEngine.from_engine_config(0, engine_config)

    task.schedule_to(engine, update_engine_info=False)

    context_mgr = ServeCoreContextManager()
    context_mgr.register_engine_prefix_cache(engine.engine_id)
    context_mgr.set_task_contexts(task)

    print(context_mgr._context_ref_counter)
    print(context_mgr.prefix_caches[engine.engine_id]._prefix_ctx_map)


if __name__ == "__main__":
    test_prefix_cache()
    test_context_manager()
