# Get FTT timepoint
import os
import re


exit_points = {}
ftt_points = {}
e2e_latency = {}
output_lens = {}

for filename in os.listdir("."):
    if filename.startswith("model_worker"):
        with open(filename, "r") as file:
            for line in file:
                match = re.search(r"hack ftt: (\d+), ([\d\-: ]+)", line)
                if match:
                    req_no = match.group(1)
                    cur_time = match.group(2)
                    ftt_points[req_no] = cur_time

        with open(filename, "r") as file:
            for line in file:
                match = re.search(r"hack request exit: (\d+), (\d+)", line)
                if match:
                    req_no = match.group(1)
                    cur_time = match.group(2)
                    exit_points[req_no] = cur_time

pattern = re.compile(r"Request (\d+): latency=([\d.]+) ms, output_len=(\d+)")
with open("vllm_chat.log", "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            request_number = match.group(1)
            latency = match.group(2)
            output_len = match.group(3)
            e2e_latency[request_number] = latency
            output_lens[request_number] = output_len

total_req_num = 40
avg_normlat = 0
avg_decode = 0

for i in range(total_req_num):
    idx = str(i)
    gen_latency = (int(exit_points[idx]) - int(ftt_points[idx])) / 1e6
    per_decode_time = gen_latency / int(output_lens[idx])
    normlat = float(e2e_latency[idx]) / int(output_lens[idx])
    avg_normlat += normlat
    avg_decode += per_decode_time
    # print("Request {}: FTT: {}, decode time: {}, gen time: {}".format(i, first_token_time, per_decode_time, gen_latency))

print("Average Normlat: ", avg_normlat / total_req_num, " ms")
print("Average decode time: ", avg_decode / total_req_num, " ms")

with open("vllm_mr.log", "r") as fp:
    for line in fp:
        match = re.search(r"Average latency: ([\d.]+) ms", line)
        if match:
            latency = float(match.group(1))
            print("MR JCT: ", latency, " ms")
