from parrot.engine.config import BuiltinConfig
from parrot.engine.builtin.builtin_runner import BuiltinRunner
from parrot.engine.primitive_job import Fill, Generate
from parrot.sampling_config import SamplingConfig


def test_simple_batch_share():
    model_name = "lmsys/vicuna-7b-v1.3"
    builtin_config = BuiltinConfig(
        num_kv_cache_blocks=1600,
        attn_func="xformers_fill_shared_prompts_generate",
        block_size=16,
    )
    runner = BuiltinRunner(model_name, builtin_config)

    batch_size = 16
    # Expect result: 32
    shared_len = 20
    diverged_lens = [i * 10 for i in range(1, batch_size + 1)]

    shared_fill = Fill(
        pid=0,
        tid=0,
        context_id=0,
        parent_context_id=-1,
        token_ids=[100] * shared_len,
    )

    diverged_fills = [
        Fill(
            pid=0,
            tid=0,
            context_id=i + 1,
            parent_context_id=0,
            token_ids=[200] * diverged_lens[i],
        )
        for i in range(batch_size)
    ]

    gens = [
        Generate(
            pid=0,
            tid=0,
            context_id=i + 1 + batch_size,
            parent_context_id=i + 1,
            sampling_config=SamplingConfig(max_gen_length=20),
        )
        for i in range(batch_size)
    ]

    runner.run_iter([shared_fill])
    runner.run_iter(diverged_fills)
    for _ in range(10):
        runner.run_iter(gens)


def test_two_level_batch_share():
    model_name = "lmsys/vicuna-7b-v1.3"
    builtin_config = BuiltinConfig(
        num_kv_cache_blocks=1600,
        attn_func="xformers_fill_shared_prompts_generate",
        block_size=16,
    )
    runner = BuiltinRunner(model_name, builtin_config)

    batch_size = 16
    # Expect result: 32+16=48
    shared_len1 = 20
    shared_len2 = 10
    diverged_lens = [i * 10 for i in range(1, batch_size + 1)]

    shared_fill1 = Fill(
        pid=0,
        tid=0,
        context_id=0,
        parent_context_id=-1,
        token_ids=[100] * shared_len1,
    )

    shared_fill2 = Fill(
        pid=0,
        tid=0,
        context_id=1,
        parent_context_id=0,
        token_ids=[100] * shared_len2,
    )

    diverged_fills = [
        Fill(
            pid=0,
            tid=0,
            context_id=i + 2,
            parent_context_id=1,
            token_ids=[200] * diverged_lens[i],
        )
        for i in range(batch_size)
    ]

    gens = [
        Generate(
            pid=0,
            tid=0,
            context_id=i + 2 + batch_size,
            parent_context_id=i + 2,
            sampling_config=SamplingConfig(max_gen_length=20),
        )
        for i in range(batch_size)
    ]

    runner.run_iter([shared_fill1])
    runner.run_iter([shared_fill2])
    runner.run_iter(diverged_fills)
    for _ in range(10):
        runner.run_iter(gens)


if __name__ == "__main__":
    # test_simple_batch_share()
    test_two_level_batch_share()
