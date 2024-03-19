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
from .nodes import BaseNode, ConstantFill, PlaceholderFill, PlaceholderGen
from .node_struct import ComputeGraph

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
}


def get_nx_graph(graph: ComputeGraph) -> nx.DiGraph:
    """Get the NetworkX graph from the StaticGraph."""

    _check_networkx()

    nx_graph = nx.DiGraph()
    for node in graph.nodes:
        nx_graph.add_node(node.short_repr())

    for node in graph.nodes:
        # Edge type A: using -
        if node.edge_a_next_node:
            nx_graph.add_edge(
                node.short_repr(), node.edge_a_next_node.short_repr(), weight=1
            )

        edge_b_next_nodes = node.edge_b_next_nodes
        for next_node in edge_b_next_nodes:
            nx_graph.add_edge(node.short_repr(), next_node.short_repr(), weight=2)

    return nx_graph


def view_graph(graph: ComputeGraph):
    """View the graph using NetworkX."""

    _check_networkx()

    nx_graph = get_nx_graph(graph)
    edge_color_list = [d["weight"] for _, _, d in nx_graph.edges(data=True)]
    edge_colors = ["red" if weight == 2 else "black" for weight in edge_color_list]

    nx.draw(
        nx_graph,
        with_labels=True,
        node_color=[_COLOR_MAP[type(node)] for node in graph.nodes],
        edge_color=edge_colors,
    )
    # plt.show()
    plt.savefig("graph.png")
