import parse
import numpy as np


total_requests = 100


with open("log/engine.log") as fp:
    lines = fp.readlines()

requests_latency = {}  # tid -> latency

for line in lines:
    result = parse.parse(
        "{prefix} Job {jobtype}(pid={pid}, tid={tid}, {misc1}) finished. Latency: {latency} ms{suffix}",
        line,
    )
    if result is not None and result["jobtype"] == "Generation":
        latency = float(result["latency"])
        tid = int(result["tid"])
        if tid not in requests_latency:
            requests_latency[tid] = 0
        requests_latency[tid] += latency
        # print(tid, latency)

requests_output_len = {}  # tid -> output_len

with open("log/os.log") as fp:
    lines = fp.readlines()

for line in lines:
    result = parse.parse(
        "{prefix}Thread - DEBUG - Thread {tid} submit Generation primitive ({misc}, max_len: {max_len}){suffix}",
        line,
    )
    if result is not None:
        tid = int(result["tid"])
        max_len = int(result["max_len"])
        requests_output_len[tid] = max_len


latencies = [requests_latency[i] for i in range(total_requests)]

# Compute the latency statistics.
avg_latency = np.mean(latencies) / 1e3
print(f"Average latency: {avg_latency:.2f} s")

per_output_latencies = [
    latencies[i] / requests_output_len[i] for i in range(total_requests)
]
avg_per_output_token_latency = np.mean(per_output_latencies)
print(
    "Average latency per output token: " f"{avg_per_output_token_latency / 1e3:.2f} s"
)

for i in range(total_requests):
    print(
        f"Request {i}: latency: {latencies[i] / 1e3:.2f} s, output len: {requests_output_len[i]}, lat_per_out: {per_output_latencies[i] / 1e3:.4f} s"
    )

# Calculate the percentage of requests which meet the latency requirement.
latency_requirement_per_output_token = 30  # 50ms, from some related posts.
num_ok_requests = [
    1 if per_output_latencies[i] <= latency_requirement_per_output_token else 0
    for i in range(total_requests)
]
print("Percentage of OK requests: " f"{np.mean(num_ok_requests):.4f}")
