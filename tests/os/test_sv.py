from parrot.os.sv.call_request import CallRequest


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
    
    parsed_request = CallRequest.parse_from_payload(payload)
    print(parsed_request)



if __name__ == "__main__":
    test_request_parse()
