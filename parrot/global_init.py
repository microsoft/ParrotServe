from .orchestration.controller import Controller
from .executor.executor import Executor
from .program.function import ParrotFunction
from .orchestration.tokenize import TokenizedStorage

parrot_global_ctrl = Controller()
global_tokenized_storage = TokenizedStorage(parrot_global_ctrl)
global_executor = Executor(parrot_global_ctrl, global_tokenized_storage)

# Set the executor
ParrotFunction._internal_executor = global_executor


def parrot_run_global():
    global_executor.run()
    parrot_global_ctrl.run()
