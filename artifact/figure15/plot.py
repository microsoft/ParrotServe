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
        method, bs, e2e, requests = tokens[0], int(tokens[1]), tokens[6], tokens[7]
        req_lat = [float(_) for _ in requests.split("+")]
        if "nan" in e2e:
            req_lat = [0]
        data[(method, bs)] = (e2e, sum(req_lat) / len(req_lat), req_lat)
    return data


data32 = read_file("result_32.txt")
data64 = read_file("result_64.txt")

olens = [200, 400, 600, 800]
systems = [
    "parrot_shared",
    "vllm_shared",
]
hatches = ["", "\\", "/", "x"]
colors = [
    "#d73027",
    # "#fee090",
    # "#91bfdb",
    "#4575b4",
]
symbols = ["o", "v"]
names = {"parrot_shared": "Parrot", "vllm_shared": "Baseline w/ Share"}


# Draw 32
x = np.arange(len(olens))
width = 0.25

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

plt.grid(True)
rects = []
for i, system in enumerate(systems):
    rect = ax.plot(
        olens,
        [data32[(system, ol)][1] / ol for ol in olens],
        markersize=10,
        marker=symbols[i],
        color=colors[i],
        label=names[system],
        zorder=3,
    )  # hatches
    rects.append(rect)

    if system == "parrot_shared":
        continue
    speedup_values = [
        data32[(system, ol)][1] / data32[("parrot_shared", ol)][1] for ol in olens
    ]
    for sid, speedup in enumerate(speedup_values):

        height = data32[(system, olens[sid])][1] / olens[sid]
        print(olens[sid], height)
        if sid == 4:
            diff = -5
        else:
            diff = 10
        ax.text(
            olens[sid] + diff,
            height + 0.01,
            "{:.2f}x".format(speedup),
            ha="center",
            va="bottom",
            rotation=70,
            fontsize=20,
        )

# plt.legend(loc='upper left', prop = { "size": 18 },)
ax.tick_params(axis="y", labelsize=20, direction="in")
ax.tick_params(axis="x", labelsize=20, direction="in")
ax.set_xlabel("Output Length (# tokens)", fontsize=20)
ax.set_ylabel("Latency per token (s)", fontsize=20)
plt.legend(loc="lower left", prop={"size": 14})  # , bbox_to_anchor= (0., 0.97))
plt.xticks(olens)
plt.yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12])
plt.ylim([0, 0.12])

plt.tight_layout()
plt.savefig("fig15_a.pdf")


# Draw 64
olens = [100, 200, 300, 400, 480]
x = np.arange(len(olens))
width = 0.25

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

plt.grid(True)
rects = []
for i, system in enumerate(systems):
    rect = ax.plot(
        olens,
        [data64[(system, ol)][1] / ol for ol in olens],
        markersize=10,
        marker=symbols[i],
        color=colors[i],
        label=names[system],
        zorder=3,
    )  # hatches
    rects.append(rect)

    if system == "parrot_shared":
        continue
    speedup_values = [
        data64[(system, ol)][1] / data64[("parrot_shared", ol)][1] for ol in olens
    ]
    for sid, speedup in enumerate(speedup_values):

        height = data64[(system, olens[sid])][1] / olens[sid]
        print(olens[sid], height)
        if sid == 4:
            diff = -5
        else:
            diff = 10
        ax.text(
            olens[sid] + diff,
            height + 0.01,
            "{:.2f}x".format(speedup),
            ha="center",
            va="bottom",
            rotation=70,
            fontsize=20,
        )

# plt.legend(loc='upper left', prop = { "size": 18 },)
ax.tick_params(axis="y", labelsize=20, direction="in")
ax.tick_params(axis="x", labelsize=20, direction="in")
ax.set_xlabel("Output Length (# tokens)", fontsize=20)
ax.set_ylabel("Latency per token (s)", fontsize=20)
plt.legend(loc="lower left", prop={"size": 14})  # , bbox_to_anchor= (0., 0.97))
plt.xticks([100, 200, 300, 400, 480])
plt.ylim([0, 0.25])

plt.tight_layout()
plt.savefig("fig15_b.pdf")
