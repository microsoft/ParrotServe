import asyncio

from parrot import P
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
        Constant,
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
            assert piece.var.is_output == expected_var_is_output[j]
            j += 1


def test_placeholder():
    a = P.placeholder("a", "a_content")
    b = P.placeholder("b")
    b.assign("b_content")

    assert a.get() == "a_content"
    assert b.get() == "b_content"


def test_placeholder_async():
    a = P.placeholder("a")

    async def main():
        a_content = a.aget()
        a.assign("a_content")
        assert await a_content == "a_content"

    asyncio.run(main())


def test_call_parrot_function():
    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    a = P.placeholder("a")
    b = P.placeholder("b")
    c = P.placeholder("c")

    test(a, b, c=c)


if __name__ == "__main__":
    test_parse_parrot_function()
    test_placeholder()
    test_placeholder_async()
    test_call_parrot_function()
