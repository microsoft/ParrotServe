import asyncio
import contextlib

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


@contextlib.contextmanager
def controller_running_context():
    """Under this context, the global controller is running."""

    global_controller.run()
    global_controller.caching_function_prefix(global_tokenized_storage)

    try:
        yield
    except BaseException as e:
        # This is mainly used to catch the error in the `main`
        #
        # For errors in coroutines, we use the fail fast mode and quit the whole system
        # In this case, we can only see a SystemExit error
        print("Error happens when executing Parrot program: ", type(e), repr(e))
        # print("Traceback: ", traceback.format_exc())

    global_controller.free_function_prefix()


def register_tokenizer(*args, **kwargs):
    global_controller.register_tokenizer(*args, **kwargs)


def register_engine(*args, **kwargs):
    global_controller.register_engine(*args, **kwargs)


def parrot_run_aysnc(coroutine):
    with controller_running_context():
        # asyncio.run(coroutine)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(coroutine)
        loop.close()
