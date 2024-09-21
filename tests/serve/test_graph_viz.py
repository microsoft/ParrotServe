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
    NativeFuncNode,
)
from parrot.serve.graph.call_request import (
    SemanticCallMetadata,
    SemanticFunctionParameter,
)
from parrot.serve.graph.visualize_utils import view_graph


def test_view_graph_simple():
    graph = ComputeGraph()

    request_chain = RequestChain.from_nodes(
        nodes=[
            ConstantFill("This is a test "),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(name="a", is_output=True)
            ),
        ]
    )

    var_mgr = SemanticVariableManager(666)
    session_id = 0
    var_mgr.register_local_var_space(session_id)
    var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain)

    graph.insert_and_update_request_chain(request_chain)

    view_graph(graph)


def test_view_graph_complex():
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

    view_graph(graph)
    activate_sv(out_var0, PerformanceCriteria.LATENCY)
    activate_sv(out_var1, PerformanceCriteria.LATENCY)
    activate_sv(out_var2, PerformanceCriteria.LATENCY)


def test_view_graph_with_native():
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
            ConstantFill("This is a test "),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(name="b", is_output=True)
            ),
        ]
    )

    var_mgr.create_vars_for_semantic_request_chain(session_id, request2)
    graph.insert_and_update_request_chain(request2)
    out_var1 = request2.comp_chains[0].gen_node.sv

    out_var2 = var_mgr.create_var(session_id=0, var_name="c")

    native_func_node = NativeFuncNode.from_variables(
        func_name="test_func",
        input_vars={"a": out_var0, "b": out_var1},
        output_vars={"c": out_var2},
    )

    graph.insert_native_func_node(native_func_node)

    request3 = RequestChain.from_nodes(
        nodes=[
            PlaceholderFill(
                parameter=SemanticFunctionParameter(
                    name="c", var_id=out_var2.id, is_output=False
                )
            ),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(name="d", is_output=True)
            ),
        ]
    )

    var_mgr.create_vars_for_semantic_request_chain(session_id, request3)
    graph.insert_and_update_request_chain(request3)

    view_graph(graph)
    # activate_sv(out_var0, PerformanceCriteria.LATENCY)
    # activate_sv(out_var1, PerformanceCriteria.LATENCY)
    activate_sv(out_var2, PerformanceCriteria.LATENCY)

    # for var in [out_var0, out_var1, out_var2]:
    #     print(var.is_activated)


if __name__ == "__main__":
    # test_view_graph_simple()
    # test_view_graph_complex()
    test_view_graph_with_native()
