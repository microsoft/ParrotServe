from parrot.engine.config import BuiltinConfig
from parrot.utils import set_random_seed
from parrot.testing.model_runner_test_template import *


def test_opt():
    set_random_seed(0)

    model_name = "facebook/opt-125m"
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


if __name__ == "__main__":
    test_opt()
