import parrot as P
from parrot.pfunc.transforms.conversation_template import vicuna_template


def test_vicuna_template():
    @P.semantic_function()
    def foo(a: P.Input, b: P.Input, c: P.Output, d: P.Output):
        """This is a test function {{a}}.
        An apple {{b}} a day keeps the doctor away.
        Please show something. {{c}}
        And something else. {{d}}
        """

    print("Before:", foo.display())

    foo = vicuna_template.transform(foo)

    print("After:", foo.display())


if __name__ == "__main__":
    test_vicuna_template()
