import pytest
import parrot as P
from parrot.program.function import Prefix, Constant, ParameterLoc


def test_parse_parrot_function():
    @P.function()
    def tell_me_a_joke(
        topic: P.Input,
        keyword: P.Input,
        joke: P.Output,
        explanation: P.Output,
    ):
        """Tell me a joke about {{topic}}. The joke must contains the following
        keywords: {{keyword}}. The following is the joke: {{joke}}. And giving a
        short explanation to show that why it is funny. The following is the explanation
        for the joke above: {{explanation}}."""

    expected_body = [
        Prefix,
        ParameterLoc,
        Constant,
        ParameterLoc,
        Constant,
        ParameterLoc,
        Constant,
        ParameterLoc,
    ]
    expected_var_is_output = [
        False,
        False,
        True,
        True,
    ]
    assert len(expected_body) == len(tell_me_a_joke.body)
    j = 0
    for i, piece in enumerate(tell_me_a_joke.body):
        assert isinstance(piece, expected_body[i])
        if isinstance(piece, ParameterLoc):
            assert piece.param.is_output == expected_var_is_output[j]
            j += 1


def test_call_function():
    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    print("Bindings:", test("a", b="b"))


def test_call_function_with_pyobjects():
    @P.function()
    def test(a: float, b: int, c: list, d: P.Output):
        """This {{b}} is a test {{a}} function {{c}} and {{d}}"""

    print(test.body)

    print("Bindings:", test(23.3, 400, [1, 2, 3, 4]))


def test_call_function_with_coroutine():
    async def get_a():
        return "a"

    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    print("Bindings:", test(get_a(), b="b"))


def test_wrongly_pass_output_argument():
    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    with pytest.raises(ValueError):
        test("a", b="b", c="c")


if __name__ == "__main__":
    test_parse_parrot_function()
    test_call_function()
    test_call_function_with_pyobjects()
    test_call_function_with_coroutine()
    test_wrongly_pass_output_argument()
