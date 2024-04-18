# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

# Simulate GPT-API latency.

import parse
import numpy


raw_data = """Sample: 0
Sorry
I
I
I
I
I
model gpt-35-turbo latency: 248.365 ms
Sample: 1
I
I
I
It
I
I
model gpt-35-turbo latency: 218.465 ms
Sample: 2
I
I
I
I
I
Can
model gpt-35-turbo latency: 232.463 ms
Sample: 3
Sorry
Sorry
I
I
I
I
model gpt-35-turbo latency: 240.661 ms
Sample: 4
I
I
I
I
I
I
model gpt-35-turbo latency: 236.236 ms
Sample: 5
I
I
Sorry
I
Sorry
I
model gpt-35-turbo latency: 239.289 ms
Sample: 6
Sorry
I
I
I
I
I
model gpt-35-turbo latency: 223.841 ms
Sample: 7
I
I
I
I
I
I
model gpt-35-turbo latency: 241.489 ms
Sample: 8
I
Can
I
Can
 I
I
model gpt-35-turbo latency: 249.026 ms
Sample: 9
I
I
I
I
I
I
model gpt-35-turbo latency: 268.284 ms
Sample: 10
Sorry
I
I
I
I
I
model gpt-35-turbo latency: 235.540 ms
Sample: 11
Sorry
I
I
I
Sorry
I
model gpt-35-turbo latency: 237.606 ms
Sample: 12
I
Can
I
I
I
I
model gpt-35-turbo latency: 237.969 ms
Sample: 13
I
I
I
I
I
I
model gpt-35-turbo latency: 249.344 ms
Sample: 14
I
I
Sorry
I
I
I
model gpt-35-turbo latency: 229.963 ms
Sample: 15
I
I
I
Sorry
I
I
model gpt-35-turbo latency: 228.231 ms
Sample: 16
I
I
I
I
I
I
model gpt-35-turbo latency: 230.732 ms
Sample: 17
I
I
Sorry
I
I
I
model gpt-35-turbo latency: 229.712 ms
Sample: 18
I
I
I
I
I
I
model gpt-35-turbo latency: 235.043 ms
Sample: 19
I
Sorry
Sorry
I
I
I
model gpt-35-turbo latency: 226.442 ms
Sample: 20
I
I
Sorry
I
Sorry
I
model gpt-35-turbo latency: 251.641 ms
Sample: 21
I
I
I
I
I
Sorry
model gpt-35-turbo latency: 234.684 ms
Sample: 22
I
Sorry
I
I
I
I
model gpt-35-turbo latency: 235.043 ms
Sample: 23
I
I
Sorry
I
I
I
model gpt-35-turbo latency: 268.978 ms
Sample: 24
I
Sorry
Sorry
I
I
I
model gpt-35-turbo latency: 316.260 ms
Sample: 25
I
I
Sorry
I
I
I
model gpt-35-turbo latency: 237.021 ms
Sample: 26
I
Can
Sorry
I
I
Sorry
model gpt-35-turbo latency: 244.932 ms
Sample: 27
Sorry
I
I
I
Sorry
I
model gpt-35-turbo latency: 231.404 ms
Sample: 28
I
I
I
I
I
Sorry
model gpt-35-turbo latency: 256.041 ms
Sample: 29
I
Ap
Sorry
I
I
I
model gpt-35-turbo latency: 235.700 ms
Sample: 30
I
Sorry
I
Sorry
Can
I
model gpt-35-turbo latency: 256.392 ms
Sample: 31
I
I
I
I
I
I
model gpt-35-turbo latency: 249.927 ms
Sample: 32
I
I
I
I
I
I
model gpt-35-turbo latency: 235.169 ms
Sample: 33
I
I
Sorry
I
I
I
model gpt-35-turbo latency: 230.860 ms
Sample: 34
Can
I
I
Sorry
I
I
model gpt-35-turbo latency: 237.695 ms
Sample: 35
Sorry
Sorry
I
I
Sorry
I
model gpt-35-turbo latency: 244.004 ms
Sample: 36
I
Can
I
I
I
I
model gpt-35-turbo latency: 238.428 ms
Sample: 37
I
Sorry
Can
I
I
I
model gpt-35-turbo latency: 248.645 ms
Sample: 38
I
I
Sorry
I
The
I
model gpt-35-turbo latency: 232.589 ms
Sample: 39
I
I
I
I
I
Sorry
model gpt-35-turbo latency: 242.137 ms
Sample: 40
Hello
I
Could
I
I
Can
model gpt-35-turbo latency: 243.390 ms
Sample: 41
Sorry
I
I
I
I
I
model gpt-35-turbo latency: 234.210 ms
Sample: 42
I
I
Could
Sorry
I
Sorry
model gpt-35-turbo latency: 233.217 ms
Sample: 43
Can
I
I
Can
I
Can
model gpt-35-turbo latency: 228.745 ms
Sample: 44
I
I
I
The
I
I
model gpt-35-turbo latency: 262.692 ms
Sample: 45
I
Can
Sorry
I
I
I
model gpt-35-turbo latency: 236.498 ms
Sample: 46
I
I
I
Sorry
I
I
model gpt-35-turbo latency: 243.124 ms
Sample: 47
I
I
I
I
I
Can
model gpt-35-turbo latency: 234.936 ms
Sample: 48
Could
Could
I
I
I
I
model gpt-35-turbo latency: 245.015 ms
Sample: 49
I
I
I
I
I
I
model gpt-35-turbo latency: 326.942 ms
Sample: 50
I
I
Sorry
I
I
I
model gpt-35-turbo latency: 265.906 ms
Sample: 51
I
I
I
Sorry
I
I
model gpt-35-turbo latency: 228.216 ms
Sample: 52
I
I
I
I
Sorry
I
model gpt-35-turbo latency: 235.843 ms
Sample: 53
Sorry
I
I
I
Sorry
Sorry
model gpt-35-turbo latency: 233.866 ms
Sample: 54
Sorry
I
Can
I
I
I
model gpt-35-turbo latency: 247.332 ms
Sample: 55
I
I
I
I
I
I
model gpt-35-turbo latency: 237.896 ms
Sample: 56
I
I
I
I
Can
I
model gpt-35-turbo latency: 246.890 ms
Sample: 57
I
I
I
I
You
It
model gpt-35-turbo latency: 234.418 ms
Sample: 58
I
Hello
I
I
Could
Could
model gpt-35-turbo latency: 234.825 ms
Sample: 59
I
Sorry
Sorry
I
Sorry
I
model gpt-35-turbo latency: 257.166 ms
Sample: 60
I
I
I
I
Sorry
I
model gpt-35-turbo latency: 229.571 ms
Sample: 61
It
I
I
I
I
I
model gpt-35-turbo latency: 231.812 ms
Sample: 62
Can
I
I
Sorry
Sorry
I
model gpt-35-turbo latency: 229.721 ms
Sample: 63
I
I
Can
I
I
I
model gpt-35-turbo latency: 234.287 ms
Sample: 64
Could
I
I
I
Sorry
Can
model gpt-35-turbo latency: 255.894 ms
Sample: 65
Sorry
I
It
I
Can
I
model gpt-35-turbo latency: 252.841 ms
Sample: 66
Can
I
I
I
I
I
model gpt-35-turbo latency: 237.878 ms
Sample: 67
I
I
I
I
I
I
model gpt-35-turbo latency: 239.173 ms
Sample: 68
I
I
Sorry
I
Sorry
I
model gpt-35-turbo latency: 242.902 ms
Sample: 69
Sorry
I
I
The
I
I
model gpt-35-turbo latency: 250.663 ms
Sample: 70
I
I
I
I
I
I
model gpt-35-turbo latency: 236.288 ms
Sample: 71
Can
Please
I
I
I
I
model gpt-35-turbo latency: 236.587 ms
Sample: 72
Sorry
I
I
Hello
I
Sorry
model gpt-35-turbo latency: 230.192 ms
Sample: 73
I
I
I
I
I
I
model gpt-35-turbo latency: 237.470 ms
Sample: 74
The
I
I



I
I
model gpt-35-turbo latency: 344.709 ms
Sample: 75
I
Can
Sorry
I
I
I
model gpt-35-turbo latency: 228.711 ms
Sample: 76
Sorry
I
I
I
I
I
model gpt-35-turbo latency: 233.952 ms
Sample: 77
I
I
I
I
I
I
model gpt-35-turbo latency: 235.648 ms
Sample: 78
I
I
There
I
I
Sorry
model gpt-35-turbo latency: 257.411 ms
Sample: 79
I
I
I
Sorry
I
I
model gpt-35-turbo latency: 238.937 ms
Sample: 80
I
I
Sorry
I
I
I
model gpt-35-turbo latency: 239.570 ms
Sample: 81
I
Sorry
I
Sorry
I
I
model gpt-35-turbo latency: 257.789 ms
Sample: 82
Sorry
I
I
I
I
I
model gpt-35-turbo latency: 233.910 ms
Sample: 83
I
It
Sorry
I
I
I
model gpt-35-turbo latency: 238.521 ms
Sample: 84
I
I
I
Sorry
I
I
model gpt-35-turbo latency: 237.783 ms
Sample: 85
I
I
Sorry
I
I
I
model gpt-35-turbo latency: 234.694 ms
Sample: 86
I
I
Sorry
I
I
I
model gpt-35-turbo latency: 233.765 ms
Sample: 87
I
Sorry
I
I
I
I
model gpt-35-turbo latency: 252.508 ms
Sample: 88
I
I
I
I
I
I
model gpt-35-turbo latency: 245.688 ms
Sample: 89
I
I
I
What
I
I
model gpt-35-turbo latency: 241.017 ms
Sample: 90
I
I
I
I
I
I
model gpt-35-turbo latency: 239.478 ms
Sample: 91
I
I
I
There
I
I
model gpt-35-turbo latency: 237.421 ms
Sample: 92
I
I
I
Hmm
I
Sorry
model gpt-35-turbo latency: 273.847 ms
Sample: 93
I
I
I
I
I
I
model gpt-35-turbo latency: 251.006 ms
Sample: 94
I
I
I
I
I
I
model gpt-35-turbo latency: 238.395 ms
Sample: 95
I
I
I
I
I
I
model gpt-35-turbo latency: 235.495 ms
Sample: 96
I
I
I
I
Sorry
I
model gpt-35-turbo latency: 238.537 ms
Sample: 97
I
I
I
I
I
I
model gpt-35-turbo latency: 231.898 ms
Sample: 98
I
I
I
Could
I
I
model gpt-35-turbo latency: 239.886 ms
Sample: 99
It
Can
I
I
I
I
model gpt-35-turbo latency: 340.099 ms
"""


def parse_latency(raw_data: str):
    lines = raw_data.split("\n")
    latencies = []
    for line in lines:
        if "latency" in line:
            latencies.append(float(parse.parse("model {} latency: {} ms", line)[1]))
    return latencies


latencies = parse_latency(raw_data)


def get_latency():
    global latencies
    return numpy.random.choice(latencies) / 1e3  # ms -> s
