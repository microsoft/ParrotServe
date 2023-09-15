from typing import List
import asyncio
from parrot.backend.engine import ExecutionEngine
from parrot.backend.backend_jobs import BackendPrimitiveJob, Fill, Generation
from parrot.utils import run_new_coro_in_current_loop
from parrot.protocol.sampling_params import SamplingParams
from transformers import AutoTokenizer


def test_engine_simple_serving():
    engine = ExecutionEngine()

    async def execute_job(job: BackendPrimitiveJob):
        engine.add_job(job)
        await job.finished.wait()

    prompt_text = "Hello, my name is"
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    prompt_tokens = tokenizer(prompt_text)["input_ids"]

    async def main():
        run_new_coro_in_current_loop(engine.execute_loop())

        print("Start")

        # Prefill
        await execute_job(
            Fill(
                session_id=0,
                context_id=0,
                parent_context_id=-1,
                token_ids=prompt_tokens,
            )
        )

        for _ in range(40):
            await execute_job(
                Generation(
                    session_id=0,
                    context_id=0,
                    sampling_params=SamplingParams(),
                )
            )
        print(tokenizer.decode(engine.runner.context_manager[0].token_ids))

    asyncio.run(main())


if __name__ == "__main__":
    test_engine_simple_serving()
