from multiprocessing import Process, Barrier
import os
import time


def start_chat_benchmark(barrier: Barrier, requests_num: int, request_rate: float):
    barrier.wait()
    os.system(
        f"""python3 benchmark_chat_serving_parrot.py \
        --num-prompts {requests_num} \
        --tokenizer hf-internal-testing/llama-tokenizer \
        --dataset ../workloads/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
        --request-rate {request_rate} \
        > parrot_chat.log"""
    )


def start_mr_benchmark(barrier: Barrier, app_num: int, app_rate: float):
    barrier.wait()

    # Chat needs some time to load ShareGPT
    time.sleep(15)

    os.system(
        f"""python3 benchmark_mr_serving_parrot.py \
        --num-apps {app_num} \
        --app-rate {app_rate} \
        > parrot_mr.log"""
    )


if __name__ == "__main__":
    barrier = Barrier(2)
    chat_proc = Process(target=start_chat_benchmark, args=(barrier, 40, 1))
    mr_proc = Process(target=start_mr_benchmark, args=(barrier, 9, 999))

    chat_proc.start()
    mr_proc.start()

    chat_proc.join()
    mr_proc.join()
