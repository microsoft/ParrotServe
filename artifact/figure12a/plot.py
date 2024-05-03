import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def read_file(filename):
    # Regex pattern to match experiment header and time
    header_pattern = r"file_name: (\w+), request_rate: (\d+\.?\d*)"
    time_pattern = r"Time: (\d+\.\d+)"

    experiments = defaultdict(list)

    # Read file
    with open(filename, "r") as f:
        lines = f.readlines()

    experiment_key = None
    for line in lines:
        header_match = re.match(header_pattern, line)
        if header_match:
            experiment_key = header_match.groups()
            # print(experiment_key)
        else:
            time_match = re.match(time_pattern, line)
            if time_match:
                experiments[experiment_key].append(float(time_match.group(1)))

    # Compute averages
    averages = {key: sum(times) / len(times) for key, times in experiments.items()}
    return averages


parrot = read_file("result_parrot.txt")
vllm = read_file("result_vllm.txt")

# print(parrot)
# print(vllm)

request_rates = ["0", "1", "2", "3", "3.5"]
systems = ["parrot", "vllm"]
hatches = ["", "\\", "/"]
symbols = ["o", "v"]
colors = ["#d73027", "#4575b4"]

# Organize the data
data = {
    "parrot": parrot,
    "vllm": vllm,
}

names = {
    "parrot": "Parrot",
    "vllm": "Baseline (vLLM)",
}

statistics = {ol: {s: [] for s in systems} for ol in request_rates}

for system, system_data in data.items():
    for key, value in system_data.items():
        request_rate = key[1]
        statistics[request_rate][system].append(value)

# Calculate statistics
averages = {
    ol: {s: np.mean(values) for s, values in ol_data.items()}
    for ol, ol_data in statistics.items()
}
averages["0"] = {"parrot": 75.42657, "vllm": 91.15404, "hf": 122.16704999999999}

# Generate the chart
x = np.arange(len(request_rates))
width = 0.25

fig, ax = plt.subplots()
max_height = 0


plt.grid(True)

for i, system in enumerate(systems):
    avg = [averages[ol][system] for ol in request_rates]
    xs = [0, 1, 2, 3, 3.5]
    rects = ax.plot(
        xs, avg, marker=symbols[i], color=colors[i], label=names[system], markersize=10
    )

    #     rects = ax.bar(x - width/2 + i*width, avg, width,  hatch = hatches[i], color = colors[i], label=names[system],zorder=3) # hatches

    # Add speedup values
    if system != "parrot":
        speedup_values = [
            averages[ol][system] / averages[ol]["parrot"] for ol in request_rates
        ]
        for _, speedup in enumerate(speedup_values):
            height = averages[request_rates[_]][system]
            max_height = max(max_height, height)
            x_diff = 0.1 if _ >= 3 else 0.2
            h_diff = -50 if _ >= 3 else 0
            ax.text(
                xs[_] + x_diff,
                height + h_diff,
                "{:.2f}x".format(speedup),
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=22,
            )

plt.legend(
    loc="upper left",
    prop={"size": 20},
)
ax.tick_params(axis="y", labelsize=20, direction="in")
ax.tick_params(axis="x", labelsize=20, direction="in")
ax.set_xlabel("Request Rate (reqs/s)", fontsize=26)
ax.set_ylabel("Average Latency (s)", fontsize=26)
# ax.set_xticks([_+0.1 for _ in x])
# ax.set_xticklabels(request_rates)
plt.xlim([-0.1, 3.8])
plt.ylim([50, max_height + 5])
ax.set_xticks([_ * 0.5 for _ in range(0, 8)])

fig.tight_layout()

plt.savefig("fig12_a.pdf")
