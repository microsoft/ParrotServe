from parrot.backend.config import NativeConfig
from model_runner_test import *


def test_opt():
    native_config = NativeConfig(
        model_name="facebook/opt-125m",
        num_kv_cache_blocks=16000,
        attn_func="xformers_with_buffer",
        random_seed=0,
    )

    test_single_fill(native_config)
    test_batch_fills(native_config)
    test_fill_then_gen(native_config)
    test_generate_single_text(native_config)
    test_generate_batch_text(native_config)
    test_fill_generate_mixed(native_config)


if __name__ == "__main__":
    test_opt()
