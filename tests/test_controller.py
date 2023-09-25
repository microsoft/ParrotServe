"""This test requires a running backend server.
Use `python3 -m parrot.testing.fake_server` to start a fake server.
"""

import time
import parrot as P
from parrot.orchestration.controller import Controller
from parrot.orchestration.tokenize import TokenizedStorage
from parrot.program.function import SemanticFunction


def test_controller_register_tokenizer():
    ctrl = Controller()
    ctrl.register_tokenizer("facebook/opt-13b")
    assert "facebook/opt-13b" in ctrl.tokenizers_table
    tokenizer = ctrl.tokenizers_table["facebook/opt-13b"]
    assert tokenizer.encode("This is a test sequence", add_special_tokens=False) == [
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
    time.sleep(6)  # wait for heartbeat


def test_controller_register_function():
    ctrl = Controller()
    tokenized_storage = TokenizedStorage(ctrl)

    ctrl.register_tokenizer("facebook/opt-13b")
    ctrl.register_engine(
        "test", host="localhost", port=8888, tokenizer="facebook/opt-13b"
    )

    SemanticFunction._controller = ctrl

    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    ctrl.run()
    ctrl.caching_function_prefix(tokenized_storage)


if __name__ == "__main__":
    test_controller_register_tokenizer()
    test_controller_register_engine()
    test_controller_register_function()
