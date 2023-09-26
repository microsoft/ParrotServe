import parrot as P
from parrot.program.transforms.prompt_formatter import (
    PyIndentRemover,
    SquashIntoOneLine,
    AlwaysOneSpace,
)


def test_py_indent_remover():
    @P.function()
    def foo(a: P.Output):
        """This is a function.
        It has multiple lines.
        And it has indents. {{a}}
        """

    print("Before:", foo.body)

    foo = PyIndentRemover().transform(foo)

    print("After:", foo.body)


def test_squash_into_one_line():
    @P.function()
    def foo(a: P.Output):
        """This
        is
        a
        function.
        It
        has multiple
        lines. {{a}}
        """

    print("Before:", foo.body)

    foo = SquashIntoOneLine().transform(foo)

    print("After:", foo.body)


def test_always_one_space():
    @P.function()
    def foo(a: P.Output):
        """This is  a   function.    It     has multiple      spaces.   {{a}}"""

    print("Before:", foo.body)

    foo = AlwaysOneSpace().transform(foo)

    print("After:", foo.body)


if __name__ == "__main__":
    test_py_indent_remover()
    test_squash_into_one_line()
    test_always_one_space()
