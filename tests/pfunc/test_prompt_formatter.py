from parrot import P
from parrot.frontend.pfunc.transforms.prompt_formatter import (
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

    print("Before:", foo.to_template_str())

    foo = PyIndentRemover().transform(foo)

    print("After:", foo.to_template_str())


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

    print("Before:", foo.to_template_str())

    foo = SquashIntoOneLine().transform(foo)

    print("After:", foo.to_template_str())


def test_always_one_space():
    @P.semantic_function(formatter=None)
    def foo(a: P.Output):
        """This is  a   function.    It     has multiple      spaces.   {{a}}"""

    print("Before:", foo.to_template_str())

    foo = AlwaysOneSpace().transform(foo)

    print("After:", foo.to_template_str())


if __name__ == "__main__":
    test_py_indent_remover()
    test_squash_into_one_line()
    test_always_one_space()
