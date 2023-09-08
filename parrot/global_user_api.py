import asyncio

from .orchestration.controller import Controller
from .executor.executor import Executor
from .program.function import ParrotFunction
from .orchestration.tokenize import TokenizedStorage

# Initialize the global components

global_controller = Controller()
global_tokenized_storage = TokenizedStorage(global_controller)
global_executor = Executor(global_controller, global_tokenized_storage)

# Set the executor
ParrotFunction._controller = global_controller
ParrotFunction._executor = global_executor


def start_controller():
    global_controller.run()
    global_controller.caching_function_prefix(global_tokenized_storage)


def register_tokenizer(*args, **kwargs):
    global_controller.register_tokenizer(*args, **kwargs)


def register_engine(*args, **kwargs):
    global_controller.register_engine(*args, **kwargs)


def parrot_run_aysnc(coroutine):
    start_controller()
    asyncio.run(coroutine)
