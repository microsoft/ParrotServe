from Parrot.parrot.engine.native.native_runner import NativeRunner
from parrot.engine.config import NativeConfig
from parrot.engine.primitive_job import Fill, Generation
from parrot.protocol.sampling_config import SamplingConfig

import numpy as np
from transformers import AutoTokenizer


def test_single_fill(model_name, native_config: NativeConfig):
    runner = NativeRunner(model_name, native_config)

    job = Fill(
        pid=0,
        tid=0,
        context_id=0,
        parent_context_id=-1,
        token_ids=np.random.randint(
            50, 10000, size=1000
        ).tolist(),  # Using too-large upper bound may cause index error in Embedding
    )

    runner.run_iter([job])


def test_batch_fills(model_name, native_config: NativeConfig):
    runner = NativeRunner(model_name, native_config)
    batch_size = 16
    jobs = [
        Fill(
            pid=0,
            tid=i,
            context_id=i,
            parent_context_id=-1,
            token_ids=np.random.randint(50, 10000, size=1000).tolist(),
        )
        for i in range(batch_size)
    ]

    runner.run_iter(jobs)


def test_fill_then_gen(model_name, native_config: NativeConfig):
    runner = NativeRunner(model_name, native_config)
    runner.run_iter(
        [
            Fill(
                pid=0,
                tid=0,
                context_id=0,
                parent_context_id=-1,
                token_ids=np.random.randint(50, 10000, size=10).tolist(),
            )
        ]
    )

    runner.run_iter(
        [
            Generation(
                pid=0,
                tid=0,
                context_id=0,
                parent_context_id=-1,
                sampling_config=SamplingConfig(),
            )
        ]
    )


def test_generate_single_text(model_name, native_config: NativeConfig):
    runner = NativeRunner(model_name, native_config)
    prompt_text = "Hello, my name is"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_tokens = tokenizer(prompt_text)["input_ids"]

    fill_job = Fill(
        pid=0,
        tid=0,
        context_id=0,
        parent_context_id=-1,
        token_ids=prompt_tokens,
    )

    runner.run_iter([fill_job])

    for _ in range(40):
        runner.run_iter(
            [
                Generation(
                    pid=0,
                    tid=0,
                    context_id=0,
                    parent_context_id=-1,
                    sampling_config=SamplingConfig(),
                )
            ]
        )
    print("Generated: ", tokenizer.decode(fill_job.context.token_ids))


def test_generate_batch_text(model_name, native_config: NativeConfig):
    runner = NativeRunner(model_name, native_config)
    prompt_text = [
        "Hello, my name is",
        "Hello, my name is",
        "Hello, my name is",
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_tokens = tokenizer(prompt_text)["input_ids"]

    # Prefill
    fills = [
        Fill(
            pid=0,
            tid=i,
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
                pid=0,
                tid=i,
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


def test_fill_generate_mixed(model_name, native_config: NativeConfig):
    runner = NativeRunner(model_name, native_config)
    prompt_text = [
        "Hello, my name is",
        "Hello, my name is",
        "Hello, my name is",
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_tokens = tokenizer(prompt_text)["input_ids"]

    # Prefill
    fills = [
        Fill(
            pid=0,
            tid=i,
            context_id=i,
            parent_context_id=-1,
            token_ids=prompt_tokens[i],
        )
        for i in range(len(prompt_tokens))
    ]

    # Generations
    gens = [
        Generation(
            pid=0,
            tid=i,
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
