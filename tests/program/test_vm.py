import parrot as P
import time
import pytest

from parrot.testing.fake_serve_core_server import TESTING_SERVER_URL
from parrot.testing.localhost_server_daemon import fake_os_server


def test_heartbeat():
    with fake_os_server():
        vm = P.VirtualMachine(os_http_addr=TESTING_SERVER_URL, mode="debug")
        time.sleep(10)  # Wait for 10 sec to see the heartbeat logs


def test_e2e():
    with fake_os_server():

        @P.semantic_function()
        def test(a: P.Input, b: P.Input, c: P.Output):
            """This {{b}} is a test {{a}} function {{c}}"""

        def main():
            c = test("a", b="b")
            print(c.get())

        vm = P.VirtualMachine(os_http_addr=TESTING_SERVER_URL, mode="debug")
        vm.run(main)


# @pytest.mark.skip(reason="Not implemented yet")
def test_vm_import():
    with fake_os_server():
        vm = P.VirtualMachine(os_http_addr=TESTING_SERVER_URL, mode="debug")
        vm.import_function(
            function_name="tell_me_a_joke",
            module_path="codelib.app.common",
        )


def test_define_func():
    with fake_os_server():
        vm = P.VirtualMachine(os_http_addr=TESTING_SERVER_URL, mode="debug")
        func = vm.define_function(
            func_name="test",
            func_body="This is a {{input}}. {{output}}",
            params=[
                P.Parameter(name="input", typ=P.ParamType.INPUT_LOC),
                P.Parameter(name="output", typ=P.ParamType.OUTPUT_LOC),
            ],
        )
        print(func.display())


if __name__ == "__main__":
    # test_heartbeat()
    # test_e2e()
    # test_vm_import()
    test_define_func()
