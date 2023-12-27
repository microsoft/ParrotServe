from parrot.engine.config import BuiltinConfig
from parrot.utils import set_random_seed
from parrot.testing.model_runner_test_template import *


def test_llama_xformers_with_buffer():
    set_random_seed(0)

    model_name = "lmsys/vicuna-7b-v1.3"
    builtin_config = BuiltinConfig(
        num_kv_cache_blocks=16000,
        attn_func="xformers_with_buffer",
    )

    template_test_single_fill(model_name, builtin_config)
    template_test_batch_fills(model_name, builtin_config)
    template_test_fill_then_gen(model_name, builtin_config)
    template_test_generate_single_text(model_name, builtin_config)
    template_test_generate_batch_text(model_name, builtin_config)
    template_test_fill_generate_mixed(model_name, builtin_config)


def test_llama_vllm():
    set_random_seed(0)

    model_name = "lmsys/vicuna-7b-v1.3"
    builtin_config = BuiltinConfig(
        num_kv_cache_blocks=1024,
        block_size=16,
        attn_func="xformers_fill_vllm_paged_attention_generate",
    )

    # test_single_fill(model_name, builtin_config)
    # test_batch_fills(model_name, builtin_config)
    # test_fill_then_gen(model_name, builtin_config)
    # test_generate_single_text(model_name, builtin_config)
    # test_generate_batch_text(model_name, builtin_config)
    template_test_fill_generate_mixed(model_name, builtin_config)


if __name__ == "__main__":
    # test_llama_xformers_with_buffer()
    test_llama_vllm()
