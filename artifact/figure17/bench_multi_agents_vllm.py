import json
import asyncio
import time
from transformers import AutoTokenizer
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

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

        ret.append(round_info)

    return ret


async def execute(workloads):
    for round_info in workloads:
        round_funcs = []
        for info in round_info:
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                max_tokens=info["output_len"],
            )
            prompt = info["shared_prompt"] + info["diverged_prompt"]
            round_funcs.append(llm.ainvoke(prompt))
        await asyncio.gather(*round_funcs)


def main(branches_num: int, cache_prefix: bool = True):
    print("branches_num: ", branches_num, flush=True)
    workloads = load_workloads(branches_num)
    st = time.perf_counter_ns()
    asyncio.run(execute(workloads))
    ed = time.perf_counter_ns()
    latency = (ed - st) / 1e9
    print(f"Time: {latency:.4f}", flush=True)


def warmup():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=5)
    llm.invoke("hello")


if __name__ == "__main__":
    warmup()

    # main(16, True)

    for bn in [4, 8, 12, 16]:
        main(bn, True)
