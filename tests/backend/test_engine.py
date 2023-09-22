import asyncio
import parrot
from parrot.backend.engine import ExecutionEngine
from parrot.backend.primitives import PrimitiveJob, Fill, Generation
from parrot.utils import create_task_in_loop
from parrot.protocol.sampling_params import SamplingParams
from transformers import AutoTokenizer


def test_engine_simple_serving():
    # The config path is relative to the package path.
    # We temporarily use this way to load the config.
    package_path = parrot.__path__[0]
    engine = ExecutionEngine(package_path + "/../configs/backend_server/opt_125m.json")

    async def execute_job(job: PrimitiveJob):
        engine.add_job(job)
        await job.finish_event.wait()

    prompt_text = "Hello, my name is"
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    prompt_tokens = tokenizer(prompt_text)["input_ids"]

    async def main():
        create_task_in_loop(engine.execute_loop())

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

        await execute_job(
            Generation(
                session_id=0,
                context_id=0,
                sampling_params=SamplingParams(
                    max_gen_length=40,
                    stop_token_ids=[tokenizer.eos_token_id],
                ),
            )
        )
        print(tokenizer.decode(engine.runner.context_manager[0].token_ids))

    try:
        asyncio.run(main(), debug=True)
    except BaseException as e:
        print("Internal error happends:", e)


if __name__ == "__main__":
    test_engine_simple_serving()
