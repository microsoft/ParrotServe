import parrot as P
from parrot.frontend.transforms.prompt_formatter import (
    PyIndentRemover,
    SquashIntoOneLine,
    AlwaysOneSpace,
)


def test_py_indent_remover():
    @P.semantic_function(formatter=None)
    def foo(a: P.Output):
        """This is a function.
        It has multiple lines.
        And it has indents. {{a}}
        """

    print("Before:", foo.display())

    foo = PyIndentRemover().transform(foo)

    print("After:", foo.display())


def test_squash_into_one_line(formatter=None):
    @P.semantic_function(formatter=None)
    def foo(a: P.Output):
        """This
        is
        a
        function.
        It
        has multiple
        lines. {{a}}
        """

    print("Before:", foo.display())

    foo = SquashIntoOneLine().transform(foo)

    print("After:", foo.display())


def test_always_one_space():
    @P.semantic_function(formatter=None)
    def foo(a: P.Output):
        """This is  a   function.    It     has multiple      spaces.   {{a}}"""

    print("Before:", foo.display())

    foo = AlwaysOneSpace().transform(foo)

    print("After:", foo.display())


if __name__ == "__main__":
    test_py_indent_remover()
    test_squash_into_one_line()
    test_always_one_space()
