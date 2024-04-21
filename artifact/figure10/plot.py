import pandas as pd
import matplotlib.pyplot as plt
import parse

symbols = [".", "x", "d", "o", "v", "^"]
colors = [
    "#d73027",
    "#fc8d59",
    "#00e290",
    "#01f4f8",
    "#91bfdb",
    "#4575b4",
]
plt.style.use("default")

# read data file
with open("result.log") as fp:
    lines = fp.readlines()
df = []
for line in lines:
    parsed = parse.parse("capacity: {cap}, reqs: {reqs}", line)
    if parsed is not None:
        cap = int(parsed["cap"])
        reqs = int(parsed["reqs"])
        df.append([reqs, cap])
    parsed = parse.parse("Mean latency: {lat} ms{suf}", line)
    if parsed is not None:
        lat = float(parsed["lat"])
        df[-1].append(lat)
    parsed = parse.parse("p90 latency: {lat} ms{suf}", line)
    if parsed is not None:
        lat = float(parsed["lat"])
        df[-1].append(lat)

# preprocess the data
data = {"reqs/s": [], "max_token": [], "Mean latency": [], "p90 latency": []}
for i in range(0, len(df)):
    data["reqs/s"].append(df[i][0])
    data["max_token"].append(df[i][1])
    data["Mean latency"].append(df[i][2])
    data["p90 latency"].append(df[i][3])
df = pd.DataFrame(data)

print(data)

# convert columns to appropriate types
df["reqs/s"] = df["reqs/s"].astype(int)
df["max_token"] = df["max_token"].astype(int)
df["Mean latency"] = df["Mean latency"].astype(float)
df["p90 latency"] = df["p90 latency"].astype(float)

# plot mean latency
fig, ax1 = plt.subplots(figsize=(5, 4))


for line_id, file_name in enumerate(df["max_token"].unique()):
    temp_df = df[df["max_token"] == file_name]
    plt.plot(
        temp_df["reqs/s"],
        temp_df["Mean latency"],
        color=colors[line_id],
        marker=symbols[line_id],
        label="Capacity=%d" % file_name,
    )

plt.plot([0, 30], [40, 40], linestyle="--", color="r")
plt.xlim([4, 26])
plt.ylim([20, 65])
ax1.tick_params(axis="y", labelsize=16, direction="in")
ax1.tick_params(axis="x", labelsize=16, direction="in")
plt.legend(
    loc="upper left",
    prop={"size": 12},
)

# plt.title('Mean Latency vs eqs/token')
plt.xlabel("Requests/s", fontsize=20)
plt.ylabel("Mean Latency (ms)", fontsize=20)
plt.grid(True)

fig.tight_layout()
plt.savefig("fig10_a.pdf")

# plot p90 latency
fig, ax2 = plt.subplots(figsize=(5, 4))
for line_id, file_name in enumerate(df["max_token"].unique()):
    temp_df = df[df["max_token"] == file_name]
    plt.plot(
        temp_df["reqs/s"],
        temp_df["p90 latency"],
        color=colors[line_id],
        marker=symbols[line_id],
        label="Capacity=%d" % file_name,
    )

ax2.tick_params(axis="y", labelsize=16, direction="in")
ax2.tick_params(axis="x", labelsize=16, direction="in")
plt.legend(
    loc="upper left",
    prop={"size": 12},
)
# plt.title('P90 Latency vs eqs/token')
plt.plot([0, 30], [40, 40], linestyle="--", color="r")
plt.xlim([4, 26])
plt.ylim([20, 65])

plt.xlabel("Requests/s", fontsize=20)
plt.ylabel("P90 Latency (ms)", fontsize=20)
plt.grid(True)

fig.tight_layout()
plt.savefig("fig10_b.pdf")
