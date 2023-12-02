import parse
import numpy as np
import matplotlib.pyplot as plt


def cal_percentage(file_name: str, requirement: float):
    with open(file_name) as fp:
        lines = fp.readlines()

    total_requests = 100
    per_output_latencies = {}

    for line in lines:
        result = parse.parse(
            "Request {tid}: latency: {latency} s, output len: {outlen}, lat_per_out: {lpo} s\n",
            line,
        )

        if result is not None:
            lpo = float(result["lpo"])
            tid = int(result["tid"])
            per_output_latencies[tid] = lpo

    # Calculate the percentage of requests which meet the latency requirement.
    latency_requirement_per_output_token = requirement

    num_ok_requests = [
        1 if per_output_latencies[i] <= latency_requirement_per_output_token else 0
        for i in range(total_requests)
    ]
    # print("Percentage of OK requests: " f"{np.mean(num_ok_requests):.4f}")
    return np.mean(num_ok_requests)


def plot():
    for bs in [6, 8, 10]:
        file_name = f"bs_{bs}.log"

        x = np.linspace(0.02, 0.05, 31)
        y = [cal_percentage(file_name, i) for i in x]
        plt.plot(x, y, label=f"bs={bs}")

    plt.xlabel("Latency requirement per output token (s)")
    plt.ylabel("Percentage of OK requests")
    plt.legend()
    plt.savefig("image.png")


if __name__ == "__main__":
    # plot()
    print(cal_percentage("25reqs/token_6144.log", 0.03))
