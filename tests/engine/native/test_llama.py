from parrot.engine.config import NativeConfig
from model_runner_test import *


def test_llama_xformers_with_buffer():
    model_name = "lmsys/vicuna-7b-v1.3"
    native_config = NativeConfig(
        num_kv_cache_blocks=16000,
        attn_func="xformers_with_buffer",
        random_seed=0,
    )

    test_single_fill(model_name, native_config)
    test_batch_fills(model_name, native_config)
    test_fill_then_gen(model_name, native_config)
    test_generate_single_text(model_name, native_config)
    test_generate_batch_text(model_name, native_config)
    test_fill_generate_mixed(model_name, native_config)


def test_llama_vllm():
    model_name = "lmsys/vicuna-7b-v1.3"
    native_config = NativeConfig(
        num_kv_cache_blocks=1024,
        block_size=16,
        attn_func="xformers_fill_vllm_paged_attention_generate",
        random_seed=0,
    )

    # test_single_fill(model_name, native_config)
    test_batch_fills(model_name, native_config)
    # test_fill_then_gen(model_name, native_config)
    # test_generate_single_text(model_name, native_config)
    # test_generate_batch_text(model_name, native_config)
    # test_fill_generate_mixed(model_name, native_config)


if __name__ == "__main__":
    # test_llama_xformers_with_buffer()
    test_llama_vllm()
