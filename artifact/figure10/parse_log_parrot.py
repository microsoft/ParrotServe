import parse
import numpy as np


def main():
    request_gen_latency = {}
    request_lens = {}

    with open("log/engine.log") as fp:
        lines = fp.readlines()

    for line in lines:
        result = parse.parse(
            "{prefix} Job Generate(pid={pid}, tid={tid}, {misc1}) finished. Latency: {latency} ms{suffix}",
            line,
        )
        if result is not None:
            task_id = int(result["tid"])
            latency = float(result["latency"])
            request_gen_latency[task_id] = latency

    with open("log/os.log") as fp:
        lines = fp.readlines()

    for line in lines:
        result = parse.parse(
            "{prefix}Thread - DEBUG - Thread {tid} (pid={pid}) submit Generation primitive ({misc}, max_len: {max_len}){suffix}",
            line,
        )
        if result is not None:
            task_id = int(result["tid"])
            glen = int(result["max_len"])
            request_lens[task_id] = glen

    # print(request_gen_latency)
    # print(request_lens)

    tpot = []

    for key in request_gen_latency.keys():
        # print(
        #     f"Task {key}: {request_gen_latency[key]:.4f} ms, {request_lens[key]} tokens, {request_gen_latency[key] / request_lens[key]:.4f} ms/token"
        # )
        tpot.append(request_gen_latency[key] / request_lens[key])

    tpot = np.array(tpot)

    print(f"Mean latency: {np.mean(tpot):.4f} ms")
    print(f"p90 latency: {np.percentile(tpot, 90):.4f} ms")


if __name__ == "__main__":
    main()
