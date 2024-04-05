from parrot.serve.graph.chunked_request import ChunkedRequest
from parrot.serve.variable_manager import SemanticVariableManager
from parrot.serve.graph import RequestChain, ComputeGraph, ConstantFill, PlaceholderGen
from parrot.serve.graph.chunked_request import RequestMetadata, RequestPlaceholder
from parrot.serve.graph.visualize_utils import view_graph


def test_request_parse():
    payload = {
        "template": "This is a test {{a}} function. {{b}}",
        "placeholders": [
            {
                "name": "a",
                "is_output": False,
                "var_id": "xxx",
            },
            {
                "name": "b",
                "is_output": True,
                "sampling_config": {
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
            },
        ],
        "session_id": "1",
        "models": ["model1", "model2"],
        "model_type": "token_id",
        "remove_pure_fill": True,
    }

    chunked_request = ChunkedRequest.parse_from_payload(payload)
    print(chunked_request)


def test_split_prefix():
    payload = {
        "template": "This is a test {{a}} function. {{b}}",
        "placeholders": [
            {
                "name": "a",
                "is_output": False,
                "var_id": "xxx",
            },
            {
                "name": "b",
                "is_output": True,
                "sampling_config": {
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
            },
        ],
        "session_id": "1",
        "models": ["model1", "model2"],
        "model_type": "token_id",
        "remove_pure_fill": True,
    }

    chunked_request = ChunkedRequest.parse_from_payload(payload)
    chunked_request.split_prefix_chunk(5)
    print(chunked_request)


def test_request_chain_print():
    request_chain = RequestChain.from_nodes(
        nodes=[
            ConstantFill("This is a test "),
            PlaceholderGen(placeholder=RequestPlaceholder(name="a", is_output=True)),
        ],
        metadata=RequestMetadata(
            session_id=0,
            models=[],
            model_type="token_id",
            remove_pure_fill=True,
        ),
    )

    print(request_chain.pretty_print())


def test_chunked_request_to_chain():
    payload = {
        "template": "This is a test {{a}} function. {{b}}",
        "placeholders": [
            {
                "name": "a",
                "is_output": False,
                "var_id": "xxx",
            },
            {
                "name": "b",
                "is_output": True,
                "sampling_config": {
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
            },
        ],
        "session_id": "1",
        "models": ["model1", "model2"],
        "model_type": "token_id",
        "remove_pure_fill": True,
    }
    chunked_request = ChunkedRequest.parse_from_payload(payload)
    request_chain = RequestChain.from_chunked_request(chunked_request)
    print(request_chain.pretty_print())


def test_graph_remove():
    graph = ComputeGraph()

    request_chain = RequestChain.from_nodes(
        nodes=[
            ConstantFill("This is a test "),
            PlaceholderGen(placeholder=RequestPlaceholder(name="a", is_output=True)),
        ],
        metadata=RequestMetadata(
            session_id=0,
            models=[],
            model_type="token_id",
            remove_pure_fill=True,
        ),
    )

    var_mgr = SemanticVariableManager()
    session_id = 0
    var_mgr.register_local_var_space(session_id)
    var_mgr.create_vars_for_request(session_id, request_chain)

    graph.insert_and_update_request_chain(request_chain)

    # for i, node in enumerate(request_chain.iter()):
    #     print(i, node)

    graph.remove_completion_chain(request_chain.completion_chains[0])

    print(graph.nodes, graph.chains)


def test_view_graph():
    graph = ComputeGraph()

    request_chain = RequestChain.from_nodes(
        nodes=[
            ConstantFill("This is a test "),
            PlaceholderGen(placeholder=RequestPlaceholder(name="a", is_output=True)),
        ],
        metadata=RequestMetadata(
            session_id=0,
            models=[],
            model_type="token_id",
            remove_pure_fill=True,
        ),
    )

    var_mgr = SemanticVariableManager()
    session_id = 0
    var_mgr.register_local_var_space(session_id)
    var_mgr.create_vars_for_request(session_id, request_chain)

    graph.insert_and_update_request_chain(request_chain)

    view_graph(graph)


if __name__ == "__main__":
    # test_request_parse()
    # test_request_chain_print()
    # test_chunked_request_to_chain()
    test_graph_remove()
    # test_view_graph()
