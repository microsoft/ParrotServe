"""This test requires a running backend fake OS server in host=localhost, port=8888.
Use `python3 -m parrot.testing.fake_os_server` to start a fake server.
"""

import parrot as P
import time

from parrot.testing.fake_os_server import TESTING_SERVER_HOST, TESTING_SERVER_PORT


SERVER_URL = f"http://{TESTING_SERVER_HOST}:{TESTING_SERVER_PORT}"


def test_heartbeat():
    vm = P.VirtualMachine(SERVER_URL)
    time.sleep(10)  # Wait for 10 sec to see the heartbeat logs


def test_e2e():
    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    def main():
        c = test("a", b="b")
        print(c.get())

    vm = P.VirtualMachine(SERVER_URL)

    vm.run(main)


if __name__ == "__main__":
    test_heartbeat()
    test_e2e()
