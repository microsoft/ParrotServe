# Benchmark Results

Network latency=250ms (If not specified)

## Data Analytics / Summarization

This benchmark contains two different styles of summarization methods:
- Chain summarization
- Map-reduce summarization

Datasets:
- Arxiv-March23
- OnlineMeeting

### Chain 1 GPU, 1 VM

Model: Llama-13B
Baseline: No pipeline, submitting requests sequentially.
Settings: Chunks num=20, per_chunk_size=650
Batch size is not important, because it's sequential.

Results (s):
- Langchain + FastChat (w/o vLLM): 21.41, 21.63, 21.34, 21.28, 21.22, 21.22, 21.20, 21.20, 21.35
- Langchain + FastChat (w/ vLLM): 15.62, 15.62, 15.69, 15.65, 15.64, 15.64, 15.65, 15.68, 15.69
- Parrot + FastChat (w/o vLLM): 21.53, 21.52, 21.23, 21.36, 21.43, 21.39, 21.43, 21.38, 21.38
- Parrot + FastChat (w/ vLLM): 16.00, 15.98, 15.98, 16.16, 16.01, 16.01, 16.00, 16.00, 16.01
- Parrot baseline: 16.04, 16.02, 16.09, 16.06, 16.03, 16.05, 16.04, 16.05, 16.05
- Parrot main: 11.43, 11.42, 11.53, 11.47, 11.49, 11.49, 11.66, 11.46, 11.47

### Chain 1 GPU, multi VMs

Baseline: No pipeline, submitting requests sequentially. No App FIFO.
Settings: Chunks num=20, per_chunk_size=650, 16 VMs send requests concurrently.
And the backend engine's max_num_batched_tokens is 2560, max_batch_size=2.


### Map-Reduce 1 GPU

GPU: A100*1
Model: Llama-13B
Baseline: Engine with 2048 max_num_batched_tokens (or 2 max_num_jobs)
Settings: Chunks num=15, per_chunk_size=1000
Ours: (Upperbound) map stage marked as 8, reduce stage marked as 2 (as baselines)

Then for baselines, it takes 15/2=8 iters to finish map stage.
Approximately, each iter's running time is 50*20ms = 1s. And there is addtional 1s for reduce.

Results (s):
- Langchain + FastChat (w/o vLLM): 31.95, 31.67, 31.76, 32.07, 32.44, 32.41, 32.20, 31.94, 32.43
- Langchain + FastChat (w/ vLLM): 23.54, 23.60, 23.54, 23.56, 23.57, 23.59, 23.59, 23.57, 23.57
- Parrot + FastChat (w/o vLLM): 33.42, 33.10, 33.70, 32.64, 3298, 31.47, 32.12, 32.64, 31.89
- Parrot + FastChat (w/ vLLM): 13.60, 13.42, 13.54, 13.55, 13.58, 13.60, 13.58, 13.59, 13.58
- Parrot baseline: 14.14, 14.12, 14.00, 13.95, 14.15, 14.12, 14.11, 14.17, 14.04
- Parrot main: 5.30, 5.45, 5.30, 5.38, 5.35, 5.32, 5.32, 5.37, 5.37

### Map-Reduce 4 GPUs

Baseline: Engine with 2048 * 4 max_num_batched_tokens (or 2 max_num_jobs)
Settings: Chunks num=30, per_chunk_size=1000
Ours: (Upperbound) map stage marked as 8, reduce stage marked as 2 (as baselines)

Then for baselines, it takes 30/8=4 iters to finish map stage.

From monitoring the log, the distribution of requests are:
1: 1 1 1 1 1 1 1 1
2: 1 1 1 1 1 1 1 1
3: 1 1 1 1 1 1 1 1
4: 1 1 1 1 1 1 1
(Total: 31=8*3+7)

04:37:03,665 first request submit
04:37:03,910 last request submit
04:37:05,077 first batch (8) fill * 1, ~1s
04:37:05,480 first batch (8) fill * 2, ~0.3s
04:37:06,953 first batch (8) generate finish, ~1.5s

The critical path are 8 requests. Theoraetically, it takes 8/2=4 iters. Each iter's running time 
is approximately 30ms * 50 = 1.5s. So the total running time is 4 * 1.5 = 6s.

But the actual running time is around 15s.

Issue: https://github.com/lm-sys/FastChat/issues/2702

Results (s):
- Langchain + FastChat (w/o vLLM): 68.32, 68.31
- Langchain + FastChat (w/ vLLM): 56.80, 56.71
- Langchain + FastChat (w/o vLLM, TP):
- Langchain + FastChat (w/ vLLM, TP):
- Parrot + FastChat (w/o vLLM): 23.57, 30.58, 25.51, 30.52, 30.5, 30.68, 23.37, 23.45, 25.28
- Parrot + FastChat (w/ vLLM): 21.13, 22.91, 22.97, 20.91, 22.68
- Parrot + FastChat (w/o vLLM, TP): 
- Parrot + FastChat (w/ vLLM, TP): 
- Parrot baseline: (14.05,) 12.66, 13.92, 13.53, 12.76, 13.07, 12.58, 12.82, 13.26, 12.80
- Parrot main: (6.72,) 5.35, 5.27, 5.41, 5.23, 5.38, 5.18, 5.19, 5.27, 5.43


## Misc (Ablation Study, etc.)

This parts contain experiments for demonstrating some settings' effects.

### Chat Serving, 1 GPU

This benchmark is to demonstrate that for latency-sensitive applications (chatting), to meet 
certain latency requirement (e.g. 20ms per generated token), how large should we set `max_batch_size` and `max_total_tokens`?

It's a non-trivial trade-off between latency and throughput. The larger the batch size is, (usually) the higher the throughput is, but the latency will be higher. (Which means QoS of every user will be worse.)

Setting: ShareGPT, 100 requests max_num_batched_tokens=2560.

We only count the model execution time, not including the queueing time, since in real case 
these requests will be rejected.

latency requirement: 30ms per generated token


Burst results (max_total_tokens, percentage of OK requests):
- 12288: 0.06
- 10240: 0.25
- 8192: 0.53
- 6144: 0.85
- 4096: 0.90
- 2048: 0.96

Serving results (25 req/s):
- 12288: 0.06
- 10240: 0.28
- 8192: 0.45
- 6144: 0.51
- 4096: 0.78
- 2048: 0.8

Hence, for meeting 30ms latency requirement, a batch of total tokens (including KV cache) should be better <= 4096.

In some of our experiments, we let baseline to use 6000~10000 tokens.