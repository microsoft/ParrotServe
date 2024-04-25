import re


def parse_req_mapping(file_path):
    pattern = r"Req mapping:\s*(\d+),\s*(\d+)"

    req_mapping = {}

    with open(file_path, "r") as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                key = int(match.group(1))  # req no
                value = int(match.group(2))  # tid
                req_mapping[key] = value

    return req_mapping  # req no -> tid


def extract_tid_if_pid_zero(file_path):
    ret = {}

    with open(file_path, "r") as file:
        for line in file:
            match = re.search(
                r"Generate stream latency: ([0-9.]+) ms\. pid=(\d+), tid=(\d+)", line
            )
            if match:
                latency = float(match.group(1))
                pid = int(match.group(2))
                tid = int(match.group(3))
                if pid == 1:
                    ret[tid] = latency

    return ret


req_mapping = parse_req_mapping("log/os_stdout.out")
# print(req_mapping)
gen_latencies = extract_tid_if_pid_zero("log/os.log")
# print(gen_latencies)

e2e_latency = {}
output_lens = {}

pattern = re.compile(r"Request (\d+): latency=([\d.]+) ms, output_len=(\d+)")
with open("parrot_chat.log", "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            request_number = int(match.group(1))
            latency = float(match.group(2))
            output_len = int(match.group(3))
            e2e_latency[request_number] = latency
            output_lens[request_number] = output_len

total_req_num = 40

avg_normlat = 0
avg_decode = 0

for i in range(total_req_num):
    tid = req_mapping[i]
    gen_latency = gen_latencies[tid]
    normlat = e2e_latency[i] / output_lens[i]
    per_decode_time = gen_latency / output_lens[i]

    # print("Request {}: FTT: {}, decode time: {}, gen time: {}".format(i, first_token_time, per_decode_time, gen_latency))

    avg_normlat += normlat
    avg_decode += per_decode_time

print(f"Average Normlat: {avg_normlat / total_req_num:.4f} ms")
print(f"Average decode time: {avg_decode / total_req_num:.4f} ms")

with open("parrot_mr.log", "r") as fp:
    for line in fp:
        match = re.search(r"Average latency: ([\d.]+) ms", line)
        if match:
            latency = float(match.group(1))
            print(f"MR JCT: {latency:.4f} ms")
