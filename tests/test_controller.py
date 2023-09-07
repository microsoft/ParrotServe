"""This test requires a running backend server.
Use `python3 -m parrot.testing.fake_server` to start a fake server.
"""

import time
from parrot.orchestration.controller import Controller


def test_controller_register_tokenizer():
    ctrl = Controller()
    ctrl.register_tokenizer("facebook/opt-13b")
    assert "facebook/opt-13b" in ctrl.tokenizers_table
    tokenizer = ctrl.tokenizers_table["facebook/opt-13b"]
    assert tokenizer("This is a test sequence")["input_ids"] == [
        2,
        713,
        16,
        10,
        1296,
        13931,
    ]


def test_controller_register_engine():
    ctrl = Controller()
    ctrl.register_tokenizer("facebook/opt-13b")
    ctrl.register_engine(
        "test", host="localhost", port=8888, tokenizer="facebook/opt-13b"
    )
    ctrl.run()
    time.sleep(10)  # wait for heartbeat


if __name__ == "__main__":
    test_controller_register_tokenizer()
    test_controller_register_engine()
