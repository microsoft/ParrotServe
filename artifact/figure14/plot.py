import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def read_file(filename):
    with open(filename, "r") as fp:
        lines = fp.readlines()
    data = {}
    for line in lines[1:]:
        tokens = line.strip().split(",")
        # print(tokens)
        method, bs, e2e, requests = tokens[0], int(tokens[1]), tokens[6], tokens[7]
        req_lat = [float(_) for _ in requests.split("+")]
        if "nan" in e2e:
            req_lat = [0]
        data[(method, bs)] = (e2e, sum(req_lat) / len(req_lat), req_lat)
    return data


data = read_file("result.txt")
data[("vllm_diverged", 32)] = data[("vllm_diverged", 64)] = ("0", 0, 0)

batch_sizes = [8, 16, 32, 64]
systems = [
    "parrot_shared",
    "vllm_shared",
    "vllm_diverged",
]
hatches = ["", "\\", "/", "x"]
colors = ["#d73027", "#fee090", "#91bfdb", "#4575b4"]


names = {
    "parrot_shared": "Parrot",
    "vllm_shared": "Baseline w/ Share",
    "vllm_diverged": "Baseline w/o Share",
}

# Generate the chart
x = np.arange(len(batch_sizes))
width = 0.25

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

plt.grid(True)
rects = []
for i, system in enumerate(systems):
    rect = ax.bar(
        x - width / 2 + i * width - 0.1,
        [data[(system, bs)][1] for bs in batch_sizes],
        width,
        hatch=hatches[i],
        color=colors[i],
        label=names[system],
        zorder=3,
    )  # hatches
    rects.append(rect)

    if system == "parrot_shared":
        continue
    speedup_values = [
        data[(system, bs)][1] / data[("parrot_shared", bs)][1] for bs in batch_sizes
    ]
    for rect, speedup in zip(rect, speedup_values):
        if speedup < 0.1:
            continue
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height,
            "{:.1f}x".format(speedup),
            ha="center",
            va="bottom",
            rotation=70,
            fontsize=20,
        )

ax.text(2.2, 0.6, "x", color="r", fontsize=30)
ax.text(3.2, 0.6, "x", color="r", fontsize=30)
# plt.legend(loc='upper left', prop = { "size": 18 },)
ax.tick_params(axis="y", labelsize=25, direction="in")
ax.tick_params(axis="x", labelsize=25, direction="in")
ax.set_xlabel("Batch Size", fontsize=25)
ax.set_ylabel("Avg. Latency (s)", fontsize=25)
ax.set_xticks([_ for _ in x])
ax.set_xticklabels(batch_sizes)
plt.legend(loc="upper left", prop={"size": 14})  # , bbox_to_anchor= (0., 0.97))
plt.ylim([0, 40])
plt.yticks([0, 10, 20, 30, 40])

plt.tight_layout()
plt.savefig("fig14.pdf")
