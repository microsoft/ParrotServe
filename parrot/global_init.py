from .orchestration.controller import Controller
from .executor.executor import Executor
from .program.function import ParrotFunction
from .orchestration.tokenize import TokenizedStorage

global_controller = Controller()
global_tokenized_storage = TokenizedStorage(global_controller)
global_executor = Executor(global_controller, global_tokenized_storage)

# Set the executor
ParrotFunction._controller = global_controller
ParrotFunction._executor = global_executor


def parrot_run_global():
    global_executor.run()
    global_controller.run()
    global_controller.caching_function_prefix(global_tokenized_storage)
