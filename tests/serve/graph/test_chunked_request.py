from parrot.serve.graph.chunked_request import ChunkedRequest


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

    parsed_request = ChunkedRequest.parse_from_payload(payload)
    print(parsed_request)


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


if __name__ == "__main__":
    # test_request_parse()
    test_split_prefix()
