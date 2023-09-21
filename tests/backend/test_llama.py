from parrot.backend.config import RunnerConfig
from model_runner_test import *


def test_llama():
    runner_config = RunnerConfig(
        model_name="lmsys/vicuna-7b-v1.3",
        num_kv_cache_blocks=16000,
        attn_func="xformers_with_buffer",
        random_seed=0,
    )

    # test_single_fill(runner_config)
    test_batch_fills(runner_config)
    # test_fill_then_gen(runner_config)
    # test_generate_single_text(runner_config)
    # test_generate_batch_text(runner_config)
    # test_fill_generate_mixed(runner_config)


if __name__ == "__main__":
    test_llama()
