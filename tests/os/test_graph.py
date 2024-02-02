from parrot.os.graph.request_parser import ParsedRequest, RequestParser


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
        "session_id": "xxx",
        "session_auth": "yyy",
        # "models": ["model1", "model2", ...], # Test default value ([])
        # "model_type": "token_id", # Test default value ("token_id")
        # "remove_pure_fill": True, # Test default value (True)
    }
    parser = RequestParser()
    parsed_request = parser.parse(payload)
    print(parsed_request.pretty_print())


if __name__ == "__main__":
    test_request_parse()
