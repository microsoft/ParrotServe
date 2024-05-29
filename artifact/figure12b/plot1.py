import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def read_file(filename):
    with open(filename, "r") as fp:
        lines = fp.readlines()

    # Define a function to parse a single block of data
    def parse_block(client_line, jct_line, avg_jct_line):
        # Extract the number of clients using regex
        clients_num_match = re.search(r"clients_num:\s*(\d+)", client_line)
        if clients_num_match:
            clients_num = int(clients_num_match.group(1))
        else:
            raise ValueError("Number of clients not found")

        # Extract the JCT dictionary
        jct_data = eval(jct_line.strip())

        # Extract the average JCT using regex
        avg_jct_match = re.search(r"Avg\. JCT ([\d.]+) \(s\)", avg_jct_line)
        if avg_jct_match:
            avg_jct = float(avg_jct_match.group(1))
        else:
            raise ValueError("Average JCT not found")

        return clients_num, jct_data, avg_jct

    # Iterate over the lines and parse each block of data
    parsed_data = []
    for i in range(0, len(lines), 3):
        clients_num, jct_data, avg_jct = parse_block(
            lines[i], lines[i + 1], lines[i + 2]
        )
        parsed_data.append(
            {"clients_num": clients_num, "jct_data": jct_data, "avg_jct": avg_jct}
        )

    # Display the parsed data
    for block in parsed_data:
        print(f"Clients: {block['clients_num']}, Average JCT: {block['avg_jct']} (s)")
    return parsed_data


parrot = read_file("result_parrot.txt")
vllm = read_file("result_vllm.txt")

systems = ["parrot", "vllm"]
hatches = ["", "\\", "/"]
colors = ["#d73027", "#4575b4"]

sys_a = parrot[-1]["jct_data"]
sys_b = vllm[-1]["jct_data"]

diff = {}
for task_id in sys_b:
    diff[task_id + 1] = sys_b[task_id] - sys_a[task_id]

fig, ax = plt.subplots(1, 1, figsize=(16, 6))
plt.grid(True)

ax.bar(
    list(diff.keys()), list(diff.values()), color=colors[0], hatch=hatches[0], zorder=3
)

ax.tick_params(axis="y", labelsize=20, direction="in")
ax.tick_params(axis="x", labelsize=20, direction="in")
ax.set_xlabel("Application No.", fontsize=20)
ax.set_ylabel("Latency in Baseline - Latency in Parrot (s)", fontsize=18)
ax.set_xticks(range(1, 26))

plt.xlim([0.3, 25.7])
plt.ylim([0, 250])

fig.tight_layout()

plt.savefig("figure13_new.pdf")
