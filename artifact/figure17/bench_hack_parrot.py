import json
import asyncio
import sys
import parse
from transformers import AutoTokenizer

import parrot as P

# HACK VERSION.
# TODO: Variable sharing between requests.


def load_workloads(branches_num: int):
    """Returns something like:

    {"shared_prompt": xxx, "diverged_prompt": xxx, "output_len": xxx}

    """

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    with open("../workloads/metagpt/log_3_round.jsonl", encoding="utf8") as f:
        log = [json.loads(line) for line in f.readlines()]

    # Replicate round
    replicate_num = 3
    replicated = []
    for _ in range(replicate_num):
        replicate_body = log[2:].copy()
        replicated.extend(replicate_body)
    log.extend(replicated)

    ret = []

    for i, round in enumerate(log):
        idx = i
        if idx > 3:
            idx = idx % 2 + 2

        queries = round[f"r{idx}_queries"]
        responses = round[f"r{idx}_responses"]

        previous_prompt = None
        shared_prompt = ""

        round_info = []

        counter = 0

        batch_sum = 0

        for kk in range(branches_num):
            keys = list(queries.keys())
            if len(keys) == 1:
                k = keys[0]
            else:
                k = keys[kk % 8]

            counter += 1

            query = queries[k]
            response = responses[k]

            prompt = query["system"] + query["user_msg"]
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

            output = response["choices"][0]["message"]["content"]
            output_len = len(tokenizer.encode(output, add_special_tokens=False))

            batch_sum += prompt_len + output_len

            if previous_prompt is None:
                previous_prompt = prompt
            elif shared_prompt == "":
                # Find common prefix.
                for j in range(min(len(prompt), len(previous_prompt))):
                    if prompt[j] != previous_prompt[j]:
                        break
                shared_prompt = prompt[:j]

            round_info.append({"output_len": output_len})

        print("batch_sum: ", batch_sum, flush=True)

        for info in round_info:
            info["shared_prompt"] = shared_prompt
            info["diverged_prompt"] = prompt[len(shared_prompt) :]
            info["shared_prompt_len"] = len(
                tokenizer.encode(shared_prompt, add_special_tokens=False)
            )
            info["diverged_prompt_len"] = len(
                tokenizer.encode(info["diverged_prompt"], add_special_tokens=False)
            )

        ret.append(round_info)

    return ret


async def execute(vm: P.VirtualMachine, workloads, cache_prefix):
    funcs = []
    for round_info in workloads:
        round_funcs = []
        for info in round_info:
            func = vm.define_function(
                func_name=None,
                func_body="1" + info["shared_prompt"] + "{{input}}{{output}}",
                params=[
                    P.Parameter(name="input", typ=P.ParamType.INPUT_LOC),
                    P.Parameter(
                        name="output",
                        typ=P.ParamType.OUTPUT_LOC,
                        sampling_config=P.SamplingConfig(
                            max_gen_length=info["output_len"],
                            ignore_tokenizer_eos=True,
                        ),
                    ),
                ],
                cache_prefix=cache_prefix,
            )
            round_funcs.append(func)
        funcs.append(round_funcs)

    for i, round_info in enumerate(workloads):
        layer_outputs = []

        vm.set_batch()
        for j, info in enumerate(round_info):
            layer_outputs.append(P.variable())
            funcs[i][j](input=info["diverged_prompt"], output=layer_outputs[j])
        await vm.submit_batch()

        # Wait for the round to finish.
        await asyncio.gather(*[output.aget() for output in layer_outputs])

        # for j, info in enumerate(round_info):
        #     string = outputs[i][j].get()
        #     print("Output: ", string)

    # inputs[0][0].set(workloads[0][0]["diverged_prompt"])
    # string = await outputs[0][0].aget()
    # print(string)


def main(branches_num: int, cache_prefix: bool = True):
    print("branches_num: ", branches_num, flush=True)
    workloads = load_workloads(branches_num)
    vm = P.VirtualMachine(os_http_addr="http://localhost:9000")
    latency = vm.run(execute, args=[vm, workloads, cache_prefix], timeit=True)
    latency -= 0.25 * 8 * 3  # Hack the communication overhead.
    print(f"Time: {latency} (s)", flush=True)

    # Browse the log to get the max allocated memory.
    max_num_tokens = 0
    with open("log/engine.log", "r") as f:
        lines = f.readlines()
        for line in lines:
            result = parse.parse(
                "{pre}num_cached_tokens: {num_tokens}",
                line,
            )
            if result is not None:
                max_num_tokens = max(max_num_tokens, int(result["num_tokens"]))
    print(f"blocks_num:  {max_num_tokens // 16}", flush=True)


def warmup():
    vm = P.VirtualMachine(os_http_addr="http://localhost:9000")
    test_func = vm.import_function(
        "func_1i_1o_genlen_100", "artifact.workloads.test_examples.normal_functions"
    )
    with vm.running_scope():
        holder = test_func("Test")
        holder.get()


if __name__ == "__main__":
    warmup()

    arg = sys.argv[1]

    if arg == "no_cache":
        # Note: 12, 16 memory full
        for bn in [4, 8]:
            main(bn, False)
    else:
        assert arg == "cache"

        for bn in [4, 8, 12, 16]:
            main(bn, True)
