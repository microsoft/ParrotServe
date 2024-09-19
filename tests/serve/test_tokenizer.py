from parrot.serve.tokenizer_wrapper import TokenizersWrapper

from parrot.serve.variable_manager import SemanticVariableManager
from parrot.serve.scheduler import CompletionTask
from parrot.sampling_config import SamplingConfig
from parrot.serve.graph import (
    RequestChain,
    ConstantFill,
    PlaceholderGen,
    PlaceholderFill,
)
from parrot.serve.graph.call_request import SemanticCallMetadata, SemanticFunctionParameter


TESTING_PROMPT_TEXT = (
    "He is widely acknowledged as one of the top achievers in his class"
)
TESTING_TOKEN_IDS = [
    940,
    338,
    17644,
    24084,
    3192,
    408,
    697,
    310,
    278,
    2246,
    3657,
    347,
    874,
    297,
    670,
    770,
]


def test_encode():
    tokenizers_wrapper = TokenizersWrapper()
    tokenizer_name = "hf-internal-testing/llama-tokenizer"
    tokenizers_wrapper.register_tokenizer(tokenizer_name)

    encoded = tokenizers_wrapper.tokenize(TESTING_PROMPT_TEXT, tokenizer_name)

    # print(encoded)
    assert encoded == TESTING_TOKEN_IDS


def test_decode():
    tokenizers_wrapper = TokenizersWrapper()
    tokenizer_name = "hf-internal-testing/llama-tokenizer"
    tokenizers_wrapper.register_tokenizer(tokenizer_name)

    decoded = tokenizers_wrapper.detokenize(TESTING_TOKEN_IDS, tokenizer_name)

    assert TESTING_PROMPT_TEXT == decoded


def test_tokenize_request():
    session_id = 0
    var_mgr = SemanticVariableManager(666)
    var_mgr.register_local_var_space(session_id=0)
    var0 = var_mgr.create_var(session_id, "a")

    request_chain = RequestChain.from_nodes(
        nodes=[
            ConstantFill("Test1"),
            PlaceholderFill(
                parameter=SemanticFunctionParameter(
                    name="a", var_id=var0.id, is_output=False
                )
            ),
            ConstantFill("Test2"),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(
                    name="b", is_output=True, sampling_config=SamplingConfig()
                )
            ),
        ]
    )

    task = CompletionTask(task_id=0, chain=request_chain.comp_chains[0])

    tokenizers_wrapper = TokenizersWrapper()
    tokenizer_name1 = "hf-internal-testing/llama-tokenizer"
    tokenizer_name2 = "facebook/opt-13b"
    tokenizers_wrapper.register_tokenizer(tokenizer_name1)
    tokenizers_wrapper.register_tokenizer(tokenizer_name2)

    var0.set("Content0")
    var_mgr.create_vars_for_request(session_id, request_chain)
    task.tokenize_chain(tokenizers_wrapper)

    print(task.tokenized_result)
    token_ids_list1 = task.tokenized_result[tokenizer_name1]
    for token_ids in token_ids_list1:
        print(tokenizers_wrapper.detokenize(token_ids, tokenizer_name1))


if __name__ == "__main__":
    # test_encode()
    # test_decode()
    test_tokenize_request()
