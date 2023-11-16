import parrot as P
import time

from parrot.testing.fake_os_server import TESTING_SERVER_URL
from parrot.testing.localhost_server_daemon import fake_os_server


def test_heartbeat():
    with fake_os_server():
        vm = P.VirtualMachine(os_http_addr=TESTING_SERVER_URL)
        time.sleep(10)  # Wait for 10 sec to see the heartbeat logs


def test_e2e():
    with fake_os_server():
        @P.function()
        def test(a: P.Input, b: P.Input, c: P.Output):
            """This {{b}} is a test {{a}} function {{c}}"""

        def main():
            c = test("a", b="b")
            print(c.get())

        vm = P.VirtualMachine(os_http_addr=TESTING_SERVER_URL)
        vm.run(main)


if __name__ == "__main__":
    test_heartbeat()
    test_e2e()
