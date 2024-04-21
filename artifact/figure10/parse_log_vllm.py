import parse
import numpy as np


def main():
    total_requests = 100

    request_first_timestamp = {}
    request_exit_timestamp = {}
    request_gen_length = {}

    with open("vllm_server.log") as fp:
        lines = fp.readlines()

    for line in lines:
        result = parse.parse(
            "[vLLM debug] Prefill finish timestamp. request_id={request_id}, t={t}",
            line,
        )
        if result is not None:
            request_id = result["request_id"]
            t = float(result["t"])
            request_first_timestamp[request_id] = t

        result = parse.parse(
            "[vLLM debug] Request exit timestamp. request_id={request_id}, t={t}, generated_len={glen}",
            line,
        )
        if result is not None:
            request_id = result["request_id"]
            t = float(result["t"])
            glen = int(result["glen"])
            request_exit_timestamp[request_id] = t
            request_gen_length[request_id] = glen

    # print(request_gen_latency)
    # print(request_lens)

    tpot = []

    for key in request_first_timestamp.keys():
        start = request_first_timestamp[key]
        end = request_exit_timestamp[key]
        gen_len = request_gen_length[key]

        tpot.append((end - start) / 1e6 / gen_len)

    tpot = np.array(tpot)

    print(f"Mean latency: {np.mean(tpot):.4f} ms")
    print(f"p90 latency: {np.percentile(tpot, 90):.4f} ms")


if __name__ == "__main__":
    main()
