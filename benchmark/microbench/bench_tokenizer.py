from transformers import AutoTokenizer
import time


def bench_tokenize_time(tokenizer_name: str):
    print("Bench tokenize. Tokenizer: ", tokenizer_name)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    workload = "This is a test sentence. " * 200
    print("Workload string length: ", len(workload))

    def encode():
        return tokenizer.encode(workload, add_special_tokens=False)

    workload_tokens = encode()
    print("Workload tokens num: ", len(workload_tokens))

    warmups = 10
    trials = 100

    for _ in range(warmups):
        encode()

    st = time.perf_counter_ns()
    for _ in range(trials):
        encode()
    ed = time.perf_counter_ns()
    print(f"Time tokenizer encode: {(ed - st) / trials / 1e6:.3f} ms")


def bench_detokenize_time(tokenizer_name: str):
    print("Bench detokenize. Tokenizer: ", tokenizer_name)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    workload = "This is a test sentence. " * 200
    workload = tokenizer.encode(workload, add_special_tokens=False)
    print("Workload token num: ", len(workload))

    def decode():
        return tokenizer.decode(
            workload,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    warmups = 10
    trials = 100

    for _ in range(warmups):
        decode()

    st = time.perf_counter_ns()
    for _ in range(trials):
        decode()
    ed = time.perf_counter_ns()
    print(f"Time tokenizer decode: {(ed - st) / trials / 1e6:.3f} ms")


if __name__ == "__main__":
    bench_tokenize_time("facebook/opt-125m")
    bench_detokenize_time("facebook/opt-125m")
    bench_tokenize_time("hf-internal-testing/llama-tokenizer")
    bench_detokenize_time("hf-internal-testing/llama-tokenizer")
