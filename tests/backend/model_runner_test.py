from parrot.backend.runner import Runner
from parrot.backend.config import RunnerConfig
from parrot.backend.iter_state import Fill, Generation
from parrot.protocol.sampling_params import SamplingParams
import numpy as np
from transformers import AutoTokenizer


def test_single_fill(runner_config: RunnerConfig):
    runner = Runner(runner_config)

    job = Fill(
        session_id=0,
        context_id=0,
        parent_context_id=-1,
        token_ids=np.random.randint(
            50, 10000, size=1000
        ).tolist(),  # Using too-large upper bound may cause index error in Embedding
    )

    runner.run_iter([job])


def test_batch_fills(runner_config: RunnerConfig):
    runner = Runner(runner_config)
    batch_size = 16
    jobs = [
        Fill(
            session_id=i,
            context_id=i,
            parent_context_id=-1,
            token_ids=np.random.randint(50, 10000, size=1000).tolist(),
        )
        for i in range(batch_size)
    ]

    runner.run_iter(jobs)


def test_fill_then_gen(runner_config: RunnerConfig):
    runner = Runner(runner_config)
    runner.run_iter(
        [
            Fill(
                session_id=0,
                context_id=0,
                parent_context_id=-1,
                token_ids=np.random.randint(50, 10000, size=10).tolist(),
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


def test_generate_single_text(runner_config: RunnerConfig):
    runner = Runner(runner_config)
    prompt_text = "Hello, my name is"
    tokenizer = AutoTokenizer.from_pretrained(runner_config.model_name)
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


def test_generate_batch_text(runner_config: RunnerConfig):
    runner = Runner(runner_config)
    prompt_text = [
        "Hello, my name is",
        "Hello, my name is",
        "Hello, my name is",
    ]
    tokenizer = AutoTokenizer.from_pretrained(runner_config.model_name)
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


def test_fill_generate_mixed(runner_config: RunnerConfig):
    runner = Runner(runner_config)
    prompt_text = [
        "Hello, my name is",
        "Hello, my name is",
        "Hello, my name is",
    ]
    tokenizer = AutoTokenizer.from_pretrained(runner_config.model_name)
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

    # Generations
    gens = [
        Generation(
            session_id=i,
            context_id=i,
            sampling_params=SamplingParams(),
        )
        for i in range(len(prompt_tokens))
    ]

    runner.run_iter([fills[0]])  # Run the first fill
    runner.run_iter([gens[0], fills[1]])  # Run the first gen and second fill
    runner.run_iter([gens[0], gens[1], fills[2]])  # Run the second gen and third fill
    # Run the gens
    for i in range(30):
        runner.run_iter([gens[0], gens[1], gens[2]])

    for i in range(len(prompt_tokens)):
        print(
            f"Prompt {i} Generated: ",
            tokenizer.decode(runner.context_manager[i].token_ids),
        )
