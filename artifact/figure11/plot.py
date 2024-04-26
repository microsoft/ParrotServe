import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def read_file(filename):
    # Regex pattern to match experiment header and time
    header_pattern = r"file_name: (\w+), chunk_size: (\d+), output_len: (\d+)"
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
        else:
            time_match = re.match(time_pattern, line)
            if time_match:
                experiments[experiment_key].append(float(time_match.group(1)))

    # Compute averages
    averages = {key: sum(times) / len(times) for key, times in experiments.items()}
    return averages


# Draw output len

# Read files
hf = read_file("result_hf_olen.txt")
vllm = read_file("result_vllm_olen.txt")
parrot = read_file("result_parrot_olen.txt")

outlens = [25, 50, 75, 100]
output_lengths = ["25", "50", "75", "100"]
systems = ["parrot", "vllm", "hf"]
hatches = ["", "\\", "/"]
colors = ["#d73027", "#fc8d59", "#4575b4"]

# Organize the data
data = {"parrot": parrot, "vllm": vllm, "hf": hf}
names = {"parrot": "Parrot", "vllm": "Baseline (vLLM)", "hf": "Baseline (HuggingFace)"}
statistics = {ol: {s: [] for s in systems} for ol in output_lengths}

for system, system_data in data.items():
    for key, value in system_data.items():
        output_len = key[2]
        statistics[output_len][system].append(value)

# Calculate statistics
averages = {
    ol: {s: np.mean(values) for s, values in ol_data.items()}
    for ol, ol_data in statistics.items()
}
mins = {
    ol: {s: np.min(values) for s, values in ol_data.items()}
    for ol, ol_data in statistics.items()
}
maxs = {
    ol: {s: np.max(values) for s, values in ol_data.items()}
    for ol, ol_data in statistics.items()
}

# Generate the chart
x = np.arange(len(output_lengths))
width = 0.25

fig, ax = plt.subplots()

plt.grid(True)

for i, system in enumerate(systems):
    avg = [averages[ol][system] for ol in output_lengths]
    min_ = [mins[ol][system] for ol in output_lengths]
    max_ = [maxs[ol][system] for ol in output_lengths]

    rects = ax.bar(
        x - width / 2 + i * width,
        avg,
        width,
        hatch=hatches[i],
        color=colors[i],
        label=names[system],
        zorder=3,
    )  # hatches

    # Add speedup values
    if system != "parrot":
        speedup_values = [
            averages[ol][system] / averages[ol]["parrot"] for ol in output_lengths
        ]
        for rect, speedup in zip(rects, speedup_values):
            height = rect.get_height()
            diff = 0.1 if system == "vllm" else -0.1
            ax.text(
                rect.get_x() + rect.get_width() / 2 - diff,
                height,
                "{:.2f}x".format(speedup),
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=16,
            )

plt.legend(
    loc="upper left",
    prop={"size": 15},
)
ax.tick_params(axis="y", labelsize=20, direction="in")
ax.tick_params(axis="x", labelsize=20, direction="in")
ax.set_xlabel("Output Length (# tokens)", fontsize=26)
ax.set_ylabel("Average Latency (s)", fontsize=26)
ax.set_xticks([_ + 0.1 for _ in x])
ax.set_xticklabels(output_lengths)
plt.ylim([0, 250])

fig.tight_layout()

plt.savefig("fig11_a.pdf")


# Draw chunk sizes

# Read files
hf = read_file("result_hf_csize.txt")
vllm = read_file("result_vllm_csize.txt")
parrot = read_file("result_parrot_csize.txt")

chunk_sizes = ["512", "1024", "1536", "2048"]
systems = ["parrot", "vllm", "hf"]
hatches = ["", "\\", "/"]
colors = ["#d73027", "#fc8d59", "#4575b4"]

# Organize the data
data = {"parrot": parrot, "vllm": vllm, "hf": hf}
names = {"parrot": "Parrot", "vllm": "Baseline (vLLM)", "hf": "Baseline (HuggingFace)"}
statistics = {ol: {s: [] for s in systems} for ol in chunk_sizes}

for system, system_data in data.items():
    for key, value in system_data.items():
        chunk_size = key[1]
        statistics[chunk_size][system].append(value)

# Calculate statistics
averages = {
    ol: {s: np.mean(values) for s, values in ol_data.items()}
    for ol, ol_data in statistics.items()
}
mins = {
    ol: {s: np.min(values) for s, values in ol_data.items()}
    for ol, ol_data in statistics.items()
}
maxs = {
    ol: {s: np.max(values) for s, values in ol_data.items()}
    for ol, ol_data in statistics.items()
}

# Generate the chart
x = np.arange(len(chunk_sizes))
width = 0.25

fig, ax = plt.subplots()


plt.grid(True)
for i, system in enumerate(systems):
    avg = [averages[ol][system] for ol in chunk_sizes]
    min_ = [mins[ol][system] for ol in chunk_sizes]
    max_ = [maxs[ol][system] for ol in chunk_sizes]

    rects = ax.bar(
        x - width / 2 + i * width,
        avg,
        width,
        hatch=hatches[i],
        color=colors[i],
        label=names[system],
        zorder=3,
    )  # hatches

    # Add speedup values
    if system != "parrot":
        speedup_values = [
            averages[ol][system] / averages[ol]["parrot"] for ol in chunk_sizes
        ]
        for rect, speedup in zip(rects, speedup_values):
            height = rect.get_height()
            diff = 0.1 if system == "vllm" else -0.1
            ax.text(
                rect.get_x() + rect.get_width() / 2 - diff,
                height,
                "{:.2f}x".format(speedup),
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=16,
            )

plt.legend(
    loc="upper right",
    prop={"size": 15},
)
ax.tick_params(axis="y", labelsize=20, direction="in")
ax.tick_params(axis="x", labelsize=20, direction="in")
ax.set_xlabel("Chunk Size (# tokens)", fontsize=26)
ax.set_ylabel("Average Latency (s)", fontsize=26)
ax.set_xticks([_ + 0.1 for _ in x])
ax.set_xticklabels(chunk_sizes)
plt.ylim([0, 280])

fig.tight_layout()

plt.savefig("fig11_b.pdf")
