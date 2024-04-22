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


data = read_file("shared_prompt_exp_1_32.csv")
