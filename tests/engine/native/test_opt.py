from parrot.engine.config import NativeConfig
from parrot.utils import set_random_seed
from model_runner_test import *


def test_opt():
    set_random_seed(0)

    model_name = "facebook/opt-125m"
    native_config = NativeConfig(
        num_kv_cache_blocks=16000,
        attn_func="xformers_with_buffer",
    )

    test_single_fill(model_name, native_config)
    test_batch_fills(model_name, native_config)
    test_fill_then_gen(model_name, native_config)
    test_generate_single_text(model_name, native_config)
    test_generate_batch_text(model_name, native_config)
    test_fill_generate_mixed(model_name, native_config)


if __name__ == "__main__":
    test_opt()
