import time

import parrot as P
from parrot.program.vm import VirtualMachine
from parrot.constants import DEFAULT_OS_URL
from parrot.testing.localhost_server_daemon import os_server, engine_server


def test_engine_register():
    with os_server():
        with engine_server(
            engine_config_name="opt-125m.json",
            wait_ready_time=2.0,
            connect_to_os=True,
        ):
            time.sleep(5)  # test heartbeat


def test_vm_register():
    with os_server():
        vm = VirtualMachine(
            os_http_addr=DEFAULT_OS_URL,
            mode="debug",
        )

        time.sleep(5)  # test heartbeat


def test_native_func_execution():
    def main():
        @P.native_function()
        def add(a: P.Input, b: P.Input) -> P.Output:
            return str(int(a) + int(b))

        a = P.variable()
        b = P.variable()
        c = add(a, b)
        d = add(c, "1")
        e = add(a, d)
        a.set("1")
        b.set("2")
        print(e.get())

    with os_server():
        vm = VirtualMachine(
            os_http_addr=DEFAULT_OS_URL,
            mode="debug",
        )

        vm.run(main)


if __name__ == "__main__":
    # test_engine_register()
    # test_vm_register()
    test_native_func_execution()
