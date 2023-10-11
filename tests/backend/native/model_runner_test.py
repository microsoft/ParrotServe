from parrot.backend.native.runner import Runner
from parrot.backend.config import NativeConfig
from parrot.backend.native.iter_state import Fill, Generation
from parrot.protocol.sampling_config import SamplingConfig

import numpy as np
from transformers import AutoTokenizer


def test_single_fill(native_config: NativeConfig):
    runner = Runner(native_config)

    job = Fill(
        client_id="test",
        session_id=0,
        context_id=0,
        parent_context_id=-1,
        token_ids=np.random.randint(
            50, 10000, size=1000
        ).tolist(),  # Using too-large upper bound may cause index error in Embedding
    )

    runner.run_iter([job])


def test_batch_fills(native_config: NativeConfig):
    runner = Runner(native_config)
    batch_size = 16
    jobs = [
        Fill(
            client_id="test",
            session_id=i,
            context_id=i,
            parent_context_id=-1,
            token_ids=np.random.randint(50, 10000, size=1000).tolist(),
        )
        for i in range(batch_size)
    ]

    runner.run_iter(jobs)


def test_fill_then_gen(native_config: NativeConfig):
    runner = Runner(native_config)
    runner.run_iter(
        [
            Fill(
                client_id="test",
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
                client_id="test",
                session_id=0,
                context_id=0,
                parent_context_id=-1,
                sampling_config=SamplingConfig(),
            )
        ]
    )


def test_generate_single_text(native_config: NativeConfig):
    runner = Runner(native_config)
    prompt_text = "Hello, my name is"
    tokenizer = AutoTokenizer.from_pretrained(native_config.model_name)
    prompt_tokens = tokenizer(prompt_text)["input_ids"]

    fill_job = Fill(
        client_id="test",
        session_id=0,
        context_id=0,
        parent_context_id=-1,
        token_ids=prompt_tokens,
    )

    runner.run_iter([fill_job])

    for _ in range(40):
        runner.run_iter(
            [
                Generation(
                    client_id="test",
                    session_id=0,
                    context_id=0,
                    parent_context_id=-1,
                    sampling_config=SamplingConfig(),
                )
            ]
        )
    print("Generated: ", tokenizer.decode(fill_job.context.token_ids))


def test_generate_batch_text(native_config: NativeConfig):
    runner = Runner(native_config)
    prompt_text = [
        "Hello, my name is",
        "Hello, my name is",
        "Hello, my name is",
    ]
    tokenizer = AutoTokenizer.from_pretrained(native_config.model_name)
    prompt_tokens = tokenizer(prompt_text)["input_ids"]

    # Prefill
    fills = [
        Fill(
            client_id="test",
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
                client_id="test",
                session_id=i,
                context_id=i,
                parent_context_id=-1,
                sampling_config=SamplingConfig(),
            )
            for i in range(len(prompt_tokens))
        ]
        runner.run_iter(gens)

    for i in range(len(prompt_tokens)):
        print(
            f"Prompt {i} Generated: ",
            tokenizer.decode(gens[i].context.token_ids),
        )


def test_fill_generate_mixed(native_config: NativeConfig):
    runner = Runner(native_config)
    prompt_text = [
        "Hello, my name is",
        "Hello, my name is",
        "Hello, my name is",
    ]
    tokenizer = AutoTokenizer.from_pretrained(native_config.model_name)
    prompt_tokens = tokenizer(prompt_text)["input_ids"]

    # Prefill
    fills = [
        Fill(
            client_id="test",
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
            client_id="test",
            session_id=i,
            context_id=i,
            parent_context_id=-1,
            sampling_config=SamplingConfig(),
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
            tokenizer.decode(gens[i].context.token_ids),
        )
