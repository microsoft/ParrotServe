Network latency=250ms (If not specified)

### Chain 1 GPU, 1 VM

Model: Llama-13B
Baseline: No pipeline, submitting requests sequentiallyS
Settings: Chunks num=20, per_chunk_size=670
Batch size is not important, because it's sequential.

- FastChat (w/o vLLM): 17.32, 17.23, 17.28, 17.50, 17.44, 17.61, 17.51, 17.25, 17.16
- FastChat (w/ vLLM): 11.26, 11.27, 11.28, 11.27, 11.29, 11.33, 11.27, 11.27, 11.26
- Parrot baseline: 11.24, 11.32, 11.25, 11.29, 11.26, 11.26, 11.26, 11.26, 11.25, 11.29
- Parrot main: 6.81, 6.83, 6.83, 6.81, 6.81, 6.81, 6.82, 6.84, 6.82, 6.87

### Chain 1 GPU, multi VMs

Baseline: No pipeline, submitting requests sequentially.
Settings: Chunks num=20, per_chunk_size=670, 16 VMs send requests concurrently.
And the backend engine's max_num_batched_tokens is 2048, max_batch_size=2.

### Chain 1 GPU, with long Constant Prefix


### Map-Reduce 1 GPU

GPU: A100*1
Model: Llama-13B
Baseline: Engine with 2048 max_num_batched_tokens (or 2 max_num_jobs)
Settings: Chunks num=15, per_chunk_size=1000
Ours: (Upperbound) map stage marked as 8, reduce stage marked as 2 (as baselines)

Then for baselines, it takes 15/2=8 iters to finish map stage.
Approximately, each iter's running time is 50*20ms = 1s. And there is addtional 1s for reduce.

Results (s):
- Langchain + FastChat (w/o vLLM): 31.67, 31.59, 31.66, 31.83, 31.70, 31.85, 31.78, 31.79, 32.54
- Langchain + FastChat (w/ vLLM): 15.27, 15.28, 15.31, 15.30, 15.28, 15.29, 15.28, 15.28, 15.29
- Parrot + FastChat (w/o vLLM): 33.42, 33.10, 33.70, 32.64, 3298, 31.47, 32.12, 32.64, 31.89
- Parrot + FastChat (w/ vLLM): 13.60, 13.42, 13.54, 13.55, 13.58, 13.60, 13.58, 13.59, 13.58
- Parrot baseline: 14.14, 14.12, 14.00, 13.95, 14.15, 14.12, 14.11, 14.17, 14.04
- Parrot main: 5.30, 5.45, 5.30,,5.38, 5.35, 5.32, 5.32, 5.37, 5.37

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


### Chatbots


### Multi-agents ReAct applications (MetaGPT,)