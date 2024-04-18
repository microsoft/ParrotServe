import time
import torch

from parrot.utils import torch_profile
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM


def greedy_sample_one(model, input_ids, attention_mask=None, past_key_values=None):
    bs, tgt_len = input_ids.shape
    if past_key_values is not None:
        _bs, _num_heads, src_len, _head_dims = past_key_values[0][0].shape
        assert bs == _bs
    else:
        src_len = 0
    if attention_mask is None:
        attention_mask = torch.ones((bs, src_len + tgt_len), device=model.device)

    torch.cuda.synchronize()
    st = time.perf_counter_ns()

    ret = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
    )

    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    print(f"Per decode time: {(ed - st) / 1e6} ms")

    return ret


def prefill(model, prompt_len):
    input_ids = torch.randint(1000, 10000, (1, prompt_len), device=model.device)
    attention_mask = torch.ones(input_ids.shape, device=model.device)
    ret = greedy_sample_one(model, input_ids, attention_mask)
    return input_ids, ret, attention_mask


def upd(ret, attention_mask):
    sampled = torch.argmax(ret.logits[:, -1, :], axis=-1)[:, None]
    past_key_values = ret.past_key_values
    attention_mask = torch.cat(
        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
    )
    # print(attention_mask.shape)
    return sampled, past_key_values, attention_mask


# @torch.no_grad()
@torch.inference_mode()
def bench_7b_model(load_model_by_fs=False):
    # model_name = "facebook/opt-125m"
    model_name = "lmsys/vicuna-7b-v1.3"
    if load_model_by_fs:
        from fastchat.model.model_adapter import VicunaAdapter

        model, _ = VicunaAdapter().load_model(model_name, {})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    # print(model) / 0

    model = model.to("cuda")

    warmups = 10
    trials = 100

    input_ids, ret, attention_mask = prefill(model, 670)
    new_input, past_key_values, attention_mask = upd(ret, attention_mask)

    for _ in range(warmups):
        ret = greedy_sample_one(model, new_input, attention_mask, past_key_values)
        new_input, past_key_values, attention_mask = upd(ret, attention_mask)

    torch.cuda.synchronize()
    st = time.perf_counter_ns()

    for _ in range(trials):
        ret = greedy_sample_one(model, new_input, attention_mask, past_key_values)
        new_input, past_key_values, attention_mask = upd(ret, attention_mask)

    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    print(f"Average time: {(ed - st) / 1e6 / trials} ms")


if __name__ == "__main__":
    # bench_7b_model()
    bench_7b_model(load_model_by_fs=True)
