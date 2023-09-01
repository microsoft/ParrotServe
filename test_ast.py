import ast
import astpretty


with open("source.py") as fp:
    py_src = fp.read()


AST = ast.parse(py_src)
astpretty.pprint(AST)
