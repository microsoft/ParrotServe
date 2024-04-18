from parrot import P

vm = P.VirtualMachine(
    core_http_addr="http://localhost:9000",
    mode="release",
)


codegen = vm.import_function(
    function_name="alex_codegen",
    module_path="app.dev",
)


def main():
    code = codegen(
        requirement="Write a Python script which can calculate the "
        "GCD (greatest common divisor) of the given two numbers."
    )
    print(code.get())


vm.run(main)
