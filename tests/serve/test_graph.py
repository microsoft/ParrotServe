from parrot.serve.graph.call_request import ChunkedSemanticCallRequest
from parrot.serve.variable_manager import SemanticVariableManager
from parrot.serve.graph import (
    RequestChain,
    ComputeGraph,
    ConstantFill,
    PlaceholderFill,
    PlaceholderGen,
    PerformanceCriteria,
    activate_sv,
)
from parrot.serve.graph.call_request import (
    SemanticCallMetadata,
    SemanticFunctionParameter,
)


def test_request_parse():
    payload = {
        "template": "This is a test {{a}} function. {{b}}",
        "parameters": [
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
        "models": ["model1", "model2"],
        "model_type": "token_id",
        "remove_pure_fill": True,
    }

    chunked_request = ChunkedSemanticCallRequest.parse_from_payload(
        request_id=0, session_id=0, payload=payload
    )
    print(chunked_request)


def test_split_prefix():
    payload = {
        "template": "This is a test {{a}} function. {{b}}",
        "parameters": [
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
        "models": ["model1", "model2"],
        "model_type": "token_id",
        "remove_pure_fill": True,
    }

    chunked_request = ChunkedSemanticCallRequest.parse_from_payload(
        request_id=0, session_id=0, payload=payload
    )
    chunked_request.split_prefix_chunk(5)
    print(chunked_request)


def test_request_chain_print():
    request_chain = RequestChain.from_nodes(
        nodes=[
            ConstantFill("This is a test "),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(name="a", is_output=True)
            ),
        ],
    )

    print(request_chain.pretty_print())


def test_chunked_request_to_chain():
    payload = {
        "template": "This is a test {{a}} function. {{b}}",
        "parameters": [
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
        "models": ["model1", "model2"],
        "model_type": "token_id",
        "remove_pure_fill": True,
    }
    chunked_request = ChunkedSemanticCallRequest.parse_from_payload(
        request_id=0, session_id=0, payload=payload
    )
    request_chain = RequestChain.from_chunked_request(chunked_request)
    print(request_chain.pretty_print())


def test_graph_remove():
    graph = ComputeGraph()

    request_chain = RequestChain.from_nodes(
        nodes=[
            ConstantFill("This is a test "),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(name="a", is_output=True)
            ),
        ],
    )

    var_mgr = SemanticVariableManager(666)
    session_id = 0
    var_mgr.register_local_var_space(session_id)
    var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain)

    graph.insert_and_update_request_chain(request_chain)

    # for i, node in enumerate(request_chain.iter()):
    #     print(i, node)

    graph.remove_completion_chain(request_chain.comp_chains[0])

    print(graph.nodes, graph.chains)


def test_graph_traverse():
    # A graph of 3 requests
    # A -> B -> C
    graph = ComputeGraph()

    var_mgr = SemanticVariableManager(666)
    session_id = 0
    var_mgr.register_local_var_space(session_id)

    request1 = RequestChain.from_nodes(
        nodes=[
            ConstantFill("This is a test "),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(name="a", is_output=True)
            ),
        ]
    )

    var_mgr.create_vars_for_semantic_request_chain(session_id, request1)
    graph.insert_and_update_request_chain(request1)
    out_var0 = request1.comp_chains[0].gen_node.sv

    request2 = RequestChain.from_nodes(
        nodes=[
            PlaceholderFill(
                parameter=SemanticFunctionParameter(
                    name="a", var_id=out_var0.id, is_output=False
                )
            ),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(name="b", is_output=True)
            ),
        ]
    )

    var_mgr.create_vars_for_semantic_request_chain(session_id, request2)
    graph.insert_and_update_request_chain(request2)
    out_var1 = request2.comp_chains[0].gen_node.sv

    request3 = RequestChain.from_nodes(
        nodes=[
            PlaceholderFill(
                parameter=SemanticFunctionParameter(
                    name="b", var_id=out_var1.id, is_output=False
                )
            ),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(name="c", is_output=True)
            ),
        ]
    )

    var_mgr.create_vars_for_semantic_request_chain(session_id, request3)
    graph.insert_and_update_request_chain(request3)
    out_var2 = request3.comp_chains[0].gen_node.sv

    # view_graph(graph)
    activate_sv(out_var0, PerformanceCriteria.LATENCY)
    activate_sv(out_var1, PerformanceCriteria.LATENCY)
    activate_sv(out_var2, PerformanceCriteria.LATENCY)

    # Expected results: A: depth 2, B: depth 1, C: depth 0
    requests = [request1, request2, request3]
    for req in requests:
        sv = req.comp_chains[0].gen_node.sv
        assert sv.is_activated
        assert sv.criteria == PerformanceCriteria.LATENCY
        print(sv.depth)


if __name__ == "__main__":
    # test_request_parse()
    # test_request_chain_print()
    # test_chunked_request_to_chain()
    # test_graph_remove()
    # test_view_graph()
    test_graph_traverse()
