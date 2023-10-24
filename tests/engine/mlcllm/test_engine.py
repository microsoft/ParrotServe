import mlc_chat  # avoid import error
import parrot
from parrot.engine.engine_creator import create_engine
from parrot.engine.primitive_job import Fill, Generation
from parrot.protocol.sampling_config import SamplingConfig


def test_engine_simple_serving():
    # The config path is relative to the package path.
    # We temporarily use this way to load the config.
    package_path = parrot.__path__[0]
    engine = create_engine(
        engine_config_path=package_path
        + "/../configs/engine/mlcllm/Llama-2-13b-chat-hf-q4f16_1-vulkan.json",
        connect_to_os=False,
    )

    prompt_text = "Hello, my name is"

    print("Start")

    # Prefill
    engine._execute_job(
        Fill(
            pid=0,
            tid=0,
            context_id=0,
            parent_context_id=-1,
            text=prompt_text,
        )
    )

    gen_job = Generation(
        pid=0,
        tid=0,
        context_id=0,
        parent_context_id=-1,
        sampling_config=SamplingConfig(
            max_gen_length=40,
            ignore_tokenizer_eos=True,
        ),
    )
    engine._execute_job(gen_job)

    print(engine.chat_module._get_message())


if __name__ == "__main__":
    test_engine_simple_serving()
