# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# This application is a simulated Linux terminal.

import parrot as P

vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="debug",
)

bash = vm.import_function("linux_terminal", "app.simulation")


def main():
    user_cmd = input("user@linux:~$ ")
    sys_output = bash(command=user_cmd)
    print(sys_output.get())


vm.run(main)
