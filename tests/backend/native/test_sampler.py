from parrot.backend.models.opt import OPTForCausalLM
from parrot.backend.config import NativeConfig
from parrot.protocol import SamplingParams
from parrot.utils import set_random_seed
from transformers import AutoConfig
import torch


def test_sampling_one_token():
    model_config = AutoConfig.from_pretrained("facebook/opt-125m")
    native_config = NativeConfig(
        model_name="facebook/opt-125m",
        num_kv_cache_blocks=1024,
        attn_func="xformers_with_buffer",
        random_seed=2333,
    )

    # Just to get the sampler
    torch.set_default_dtype(torch.float16)
    model = OPTForCausalLM(model_config, native_config)
    model.load_weights("facebook/opt-125m")
    model = model.cuda()

    sampler = model.sampler
    set_random_seed(2333)
    hidden_states = torch.randn(
        (8, model_config.hidden_size), dtype=torch.float16, device="cuda"
    )
    ids = sampler(hidden_states[-1:], [SamplingParams()])

    assert ids[0] == 14836


if __name__ == "__main__":
    test_sampling_one_token()
