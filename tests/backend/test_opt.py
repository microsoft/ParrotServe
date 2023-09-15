from parrot.backend.runner import Runner
from parrot.backend.config import RunnerConfig
from parrot.backend.iter_state import Fill, Generation
from parrot.protocol.sampling_params import SamplingParams
import numpy as np
from transformers import AutoTokenizer


def _get_runner():
    runner_config = RunnerConfig(
        model_name="facebook/opt-125m",
        num_kv_cache_blocks=1024,
        attn_func="xformers_with_buffer",
        random_seed=0,
    )
    return Runner(runner_config)


def test_single_fill():
    runner = _get_runner()

    job = Fill(
        session_id=0,
        context_id=0,
        parent_context_id=-1,
        token_ids=np.random.randint(50, 50000, size=10).tolist(),
    )

    runner.run_iter([job])


def test_batch_fills():
    runner = _get_runner()

    jobs = [
        Fill(
            session_id=0,
            context_id=0,
            parent_context_id=-1,
            token_ids=np.random.randint(50, 50000, size=10).tolist(),
        ),
        Fill(
            session_id=1,
            context_id=1,
            parent_context_id=-1,
            token_ids=np.random.randint(50, 50000, size=15).tolist(),
        ),
        Fill(
            session_id=2,
            context_id=2,
            parent_context_id=-1,
            token_ids=np.random.randint(50, 50000, size=20).tolist(),
        ),
    ]

    runner.run_iter(jobs)


def test_fill_then_gen():
    runner = _get_runner()

    runner.run_iter(
        [
            Fill(
                session_id=0,
                context_id=0,
                parent_context_id=-1,
                token_ids=np.random.randint(50, 50000, size=10).tolist(),
            )
        ]
    )

    runner.run_iter(
        [
            Generation(
                session_id=0,
                context_id=0,
                sampling_params=SamplingParams(),
            )
        ]
    )


def test_generate_single_text():
    runner = _get_runner()
    prompt_text = "Hello, my name is"
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    prompt_tokens = tokenizer(prompt_text)["input_ids"]

    runner.run_iter(
        [
            Fill(
                session_id=0,
                context_id=0,
                parent_context_id=-1,
                token_ids=prompt_tokens,
            )
        ]
    )

    for _ in range(40):
        runner.run_iter(
            [
                Generation(
                    session_id=0,
                    context_id=0,
                    sampling_params=SamplingParams(),
                )
            ]
        )
    print("Generated: ", tokenizer.decode(runner.context_manager[0].token_ids))


def test_generate_batch_text():
    runner = _get_runner()
    prompt_text = [
        "Hello, my name is",
        "Hello, my name is",
        "Hello, my name is",
    ]
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    prompt_tokens = tokenizer(prompt_text)["input_ids"]

    # Prefill
    fills = [
        Fill(
            session_id=i,
            context_id=i,
            parent_context_id=-1,
            token_ids=prompt_tokens[i],
        )
        for i in range(len(prompt_tokens))
    ]
    runner.run_iter(fills)

    for _ in range(40):
        gens = [
            Generation(
                session_id=i,
                context_id=i,
                sampling_params=SamplingParams(),
            )
            for i in range(len(prompt_tokens))
        ]
        runner.run_iter(gens)

    for i in range(len(prompt_tokens)):
        print(
            f"Prompt {i} Generated: ",
            tokenizer.decode(runner.context_manager[i].token_ids),
        )


if __name__ == "__main__":
    test_single_fill()
    test_batch_fills()
    test_fill_then_gen()
    test_generate_single_text()
    test_generate_batch_text()
