import time
import json
import openai

from parrot.testing.get_configs import get_engine_config_path
from parrot.engine.config import OpenAIConfig
from parrot.constants import DEFAULT_ENGINE_URL


def test_azure_openai_url_latency():
    openai_config_path = get_engine_config_path("azure-openai-gpt-3.5-turbo.json")

    with open(openai_config_path, "r") as f:
        openai_engine_config_dict = json.load(f)
    openai_config = OpenAIConfig(**openai_engine_config_dict["instance"])

    assert openai_config.is_azure

    client = openai.AzureOpenAI(
        api_key=openai_config.api_key,
        api_version=openai_config.azure_api_version,
        azure_endpoint=openai_config.azure_endpoint,
    )

    st = time.perf_counter_ns()
    client.chat.completions.create(
        messages=[{"role": "user", "content": "1"}],
        model=openai_engine_config_dict["model"],
        # seed=self.engine_config.random_seed,
        max_tokens=1,
    )
    ed = time.perf_counter_ns()
    print(f"Latency: {(ed - st) / 1e6} ms")


if __name__ == "__main__":
    test_azure_openai_url_latency()
