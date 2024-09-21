# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Visualize parrot ComputeGraph."""

NETWORKX_INSTALLED = False
try:
    import networkx as nx
except ImportError:
    NETWORKX_INSTALLED = True
from matplotlib import pyplot as plt


# from parrot.utils import get_logger

from .semantic_variable import SemanticVariable
from .nodes import (
    SemanticNode,
    ConstantFill,
    PlaceholderFill,
    PlaceholderGen,
    NativeFuncNode,
)
from .graph import ComputeGraph

# logger = get_logger("GraphViz")


def _check_networkx():
    if NETWORKX_INSTALLED:
        raise ImportError(
            "NetworkX is not installed. Please install it first to enable the visualization."
        )


_COLOR_MAP = {
    ConstantFill: "gray",
    PlaceholderFill: "green",
    PlaceholderGen: "orange",
    SemanticVariable: "purple",
    NativeFuncNode: "blue",
}


def get_nx_graph(graph: ComputeGraph) -> nx.DiGraph:
    """Get the NetworkX graph from the StaticGraph."""

    _check_networkx()

    nx_graph = nx.DiGraph()
    for node in graph.nodes:
        nx_graph.add_node(node.short_repr())

    for node in graph.nodes:
        if isinstance(node, SemanticNode):
            # Edge type A: using -
            if node.has_edge_a_next_node:
                nx_graph.add_edge(
                    node.short_repr(),
                    node.get_edge_a_next_node().short_repr(),
                    weight=1,
                )

            # Edge type B
            edge_b_next_nodes = node.get_edge_b_next_nodes()
            for next_node in edge_b_next_nodes:
                nx_graph.add_edge(node.short_repr(), next_node.short_repr(), weight=2)
        else:
            # Native func node
            producers = node.get_prev_producers()
            consumers = node.get_next_consumers()
            for producer in producers:
                nx_graph.add_edge(producer.short_repr(), node.short_repr(), weight=3)
            for consumer in consumers:
                nx_graph.add_edge(node.short_repr(), consumer.short_repr(), weight=3)

    return nx_graph


def view_graph(graph: ComputeGraph):
    """View the graph using NetworkX."""

    _check_networkx()

    nx_graph = get_nx_graph(graph)
    edge_color_list = [d["weight"] for _, _, d in nx_graph.edges(data=True)]
    edge_colors = [
        "red" if weight == 2 else "gray" if weight == 3 else "black"
        for weight in edge_color_list
    ]

    nx.draw(
        nx_graph,
        with_labels=True,
        node_color=[_COLOR_MAP[type(node)] for node in graph.nodes],
        edge_color=edge_colors,
    )
    # plt.show()
    plt.savefig("graph.png")
