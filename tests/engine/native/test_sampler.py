from transformers import AutoConfig
import torch


from parrot.engine.builtin.models.opt import OPTForCausalLM
from parrot.engine.config import BuiltinConfig
from parrot.protocol.sampling_config import SamplingConfig
from parrot.utils import set_random_seed


def test_sampling_one_token():
    set_random_seed(2333)

    model_config = AutoConfig.from_pretrained("facebook/opt-125m")
    builtin_config = BuiltinConfig(
        num_kv_cache_blocks=1024, attn_func="xformers_with_buffer"
    )

    # Just to get the sampler
    torch.set_default_dtype(torch.float16)
    model = OPTForCausalLM(model_config, builtin_config)
    model.load_weights("facebook/opt-125m")
    model = model.cuda()

    sampler = model.sampler
    set_random_seed(2333)
    hidden_states = torch.randn(
        (8, model_config.hidden_size), dtype=torch.float16, device="cuda"
    )
    ids = sampler(
        hidden_states[-1:],
        [
            SamplingConfig(
                temperature=1.0,
                top_p=1.0,
            )
        ],
    )

    assert ids[0] == 14836


if __name__ == "__main__":
    test_sampling_one_token()
