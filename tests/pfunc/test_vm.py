import time
import pytest

from parrot import P

from parrot.testing.fake_core_server import TESTING_SERVER_URL
from parrot.testing.localhost_server_daemon import fake_core_server


def test_e2e():
    with fake_core_server():

        @P.semantic_function()
        def test(a: P.Input, b: P.Input, c: P.Output):
            """This {{b}} is a test {{a}} function {{c}}"""

        def main():
            c = test("a", b="b")
            print(c.get(P.PerformanceCriteria.LATENCY))

        vm = P.VirtualMachine(core_http_addr=TESTING_SERVER_URL, mode="debug")
        vm.run(main)


# @pytest.mark.skip(reason="Not implemented yet")
def test_vm_import():
    with fake_core_server():
        vm = P.VirtualMachine(core_http_addr=TESTING_SERVER_URL, mode="debug")
        vm.import_function(
            function_name="tell_me_a_joke",
            module_path="examples.codelib.app.common",
        )


def test_define_func():
    with fake_core_server():
        vm = P.VirtualMachine(core_http_addr=TESTING_SERVER_URL, mode="debug")
        func = vm.define_function(
            func_name="test",
            func_body="This is a {{input}}. {{output}}",
            params=[
                P.Parameter(name="input", typ=P.ParamType.INPUT_LOC),
                P.Parameter(name="output", typ=P.ParamType.OUTPUT_LOC),
            ],
        )
        print(func.to_template_str())


if __name__ == "__main__":
    test_e2e()
    # test_vm_import()
    # test_define_func()
