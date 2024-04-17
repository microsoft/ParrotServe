from parrot import P
from parrot.frontend.pfunc.transforms.conversation_template import vicuna_template


def test_vicuna_template():
    @P.semantic_function()
    def foo(a: P.Input, b: P.Input, c: P.Output, d: P.Output):
        """This is a test function {{a}}.
        An apple {{b}} a day keeps the doctor away.
        Please show something. {{c}}
        And something else. {{d}}
        """

    print("Before:", foo.to_template_str())

    foo = vicuna_template.transform(foo)

    print("After:", foo.to_template_str())


if __name__ == "__main__":
    test_vicuna_template()
