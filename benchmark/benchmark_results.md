### Map-Reduce 1 GPU

Baseline: Engine with 2048 max_num_batched_tokens (or 2 max_num_jobs)
Settings: Chunks num=15, per_chunk_size=1000
Ours: (Upperbound) map stage marked as 8, reduce stage marked as 2 (as baselines)

Then for baselines, it takes 15/2=8 iters to finish map stage.

Results (s):
- Langchain + FastChat (w/o vLLM): 37.99
- Langchain + FastChat (w/ vLLM): 30.18
- Parrot baseline: 23.31
- Parrot main: 15.77


### Map-Reduce 4 GPUs

Baseline: Engine with 2048 * 4 max_num_batched_tokens (or 2 max_num_jobs)
Settings: Chunks num=30, per_chunk_size=1000
Ours: (Upperbound) map stage marked as 8, reduce stage marked as 2 (as baselines)

Then for baselines, it takes 30/8=4 iters to finish map stage.

Results (s):
- Langchain + FastChat (w/o vLLM): 37.99
- Langchain + FastChat (w/ vLLM): 30.18
- Parrot baseline: 23.31
- Parrot main: 15.77