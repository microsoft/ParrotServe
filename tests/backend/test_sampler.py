from parrot.backend.models.opt import OPTForCausalLM
from parrot.backend.config import BackendConfig
from parrot.backend.iter_state import IterationState
from parrot.utils import set_random_seed
from transformers import AutoConfig
import torch


def test_sampling_one_token():
    model_config = AutoConfig.from_pretrained("facebook/opt-125m")
    attn_config = BackendConfig(
        cache_blocks_num=1024,
        attn_func="xformers_with_buffer",
        seed=2333,
    )

    # Just to get the sampler
    torch.set_default_dtype(torch.float16)
    model = OPTForCausalLM(model_config, attn_config)
    model.load_weights("facebook/opt-125m")
    model = model.cuda()

    sampler = model.sampler
    set_random_seed(2333)
    hidden_states = torch.randn(
        (8, model_config.hidden_size), dtype=torch.float16, device="cuda"
    )
    iter_state = IterationState(
        [], {}, model_config, attn_config, torch.float16, "cuda"
    )
    iter_state.num_fill_tokens = [8]
    ids = sampler(hidden_states, iter_state)

    assert ids[0] == 14836


if __name__ == "__main__":
    test_sampling_one_token()
