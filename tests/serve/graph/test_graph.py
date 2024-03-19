from parrot.serve.sv.request_parser import ParsedRequest, RequestParser, ComputeGraph
from parrot.serve.graph.visualize_utils import view_graph


def test_request_parse():
    payload = {
        "template": "This is a test {{a}} function. {{b}}",
        "placeholders": [
            {
                "name": "a",
                "is_output": False,
                "value_type": "constant",
                "const_value": "xxx",
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
    parser = RequestParser()
    parsed_request = parser.parse(payload)
    print(parsed_request.pretty_print())


def test_graph():
    parser = RequestParser()
    graph = ComputeGraph()

    payload1 = {
        "template": "This is a test {{a}} function. {{b}}",
        "placeholders": [
            {
                "name": "a",
                "is_output": False,
                "value_type": "constant",
                "const_value": "xxx",
            },
            {
                "name": "b",
                "is_output": True,
            },
        ],
    }

    parsed1 = parser.parse(payload1)
    created_svs1 = parsed1.insert_to_graph(graph)

    b_var_id = created_svs1[1]["var_id"]

    payload2 = {
        "template": "This is a test {{a}} function. {{b}}",
        "placeholders": [
            {
                "name": "a",
                "is_output": False,
                "value_type": "variable",
                "var_id": b_var_id,
            },
            {
                "name": "b",
                "is_output": True,
            },
        ],
    }

    parsed2 = parser.parse(payload2)
    created_svs2 = parsed2.insert_to_graph(graph)
    print(created_svs2)

    view_graph(graph)


if __name__ == "__main__":
    # test_request_parse()
    test_graph()
