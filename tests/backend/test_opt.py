from parrot.backend.runner import Runner
from parrot.backend.iter_state import Fill, Generation
from parrot.protocol.sampling_params import SamplingParams
import numpy as np
from transformers import AutoTokenizer


def test_single_fill():
    runner = Runner("facebook/opt-125m")

    job = Fill(
        session_id=0,
        context_id=0,
        parent_context_id=-1,
        tokens_id=np.random.randint(50, 50000, size=10).tolist(),
    )

    runner.run_iter([job])


def test_batch_fills():
    runner = Runner("facebook/opt-125m")

    jobs = [
        Fill(
            session_id=0,
            context_id=0,
            parent_context_id=-1,
            tokens_id=np.random.randint(50, 50000, size=10).tolist(),
        ),
        Fill(
            session_id=1,
            context_id=1,
            parent_context_id=-1,
            tokens_id=np.random.randint(50, 50000, size=15).tolist(),
        ),
        Fill(
            session_id=2,
            context_id=2,
            parent_context_id=-1,
            tokens_id=np.random.randint(50, 50000, size=20).tolist(),
        ),
    ]

    runner.run_iter(jobs)


def test_fill_then_gen():
    runner = Runner("facebook/opt-125m")

    runner.run_iter(
        [
            Fill(
                session_id=0,
                context_id=0,
                parent_context_id=-1,
                tokens_id=np.random.randint(50, 50000, size=10).tolist(),
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


def test_generate_text():
    runner = Runner("facebook/opt-125m")
    prompt_text = "Hello, my name is"
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    prompt_tokens = tokenizer.encode(prompt_text, return_tensors="pt")[0].tolist()

    runner.run_iter(
        [
            Fill(
                session_id=0,
                context_id=0,
                parent_context_id=-1,
                tokens_id=prompt_tokens,
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
    print(tokenizer.decode(runner.context_manager[0].tokens_id))


if __name__ == "__main__":
    # test_single_fill()
    # test_batch_fills()
    # test_fill_then_gen()
    test_generate_text()
