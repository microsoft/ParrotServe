# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Dict

from parrot.utils import get_logger, create_task_in_loop
from parrot.exceptions import parrot_assert
from parrot.protocol.internal.primitive_request import Primitive, Fill, Generate
from parrot.protocol.internal.layer_apis import FillResponse, GenerateResponse

from parrot.serve.graph import (
    ComputeGraph,
    RequestChain,
    CompletionChain,
)
from parrot.serve.scheduler import (
    CompletionTask,
    TaskCreator,
    GlobalScheduler,
    TaskStatus,
)
from parrot.serve.backend_repr import ModelType

from ..context_manager import ServeCoreContextManager
from ..engine_manager import EngineManager
from ..tokenizer_wrapper import TokenizersWrapper


logger = get_logger("GraphExecutor")


class GraphExecutor:
    """
    GraphExecutor in a session polls CompletionChain to GlobalScheduler,
    waiting for scheduling and execute it.
    """

    def __init__(
        self,
        session_id: int,
        task_creator: TaskCreator,
        scheduler: GlobalScheduler,
        engine_mgr: EngineManager,
        context_mgr: ServeCoreContextManager,
        tokenizers_wrapper: TokenizersWrapper,
    ):
        # ---------- Basic Info ----------
        self.session_id = session_id
        self.graph = ComputeGraph()

        # ---------- Global Components ----------
        self.task_creator = task_creator
        self.scheduler = scheduler
        self.engine_mgr = engine_mgr
        self.context_mgr = context_mgr
        self.tokenizers_wrapper = tokenizers_wrapper

        # ---------- Runtime ----------
        self.bad_exception: Optional[Exception] = None

    async def _execute_coroutine(self, completion_chain: CompletionChain) -> None:
        """Coroutine for executing a CompletionChain."""

        try:
            # Block until it's activated by a GET.
            await completion_chain.wait_activated()

            # Create a task object for the completion chain.
            task = self.task_creator.create_task(completion_chain)

            # Block until all inputs are ready.
            for node in completion_chain.iter_fill():
                await node.wait_ready()

            # Tokenize the task.
            task.tokenize_chain(self.tokenizers_wrapper)

            # Submit the task to the scheduler and wait for the task to be scheduled.
            self.scheduler.submit_task(task)
            await task.wait_scheduled()
        except Exception as e:
            logger.error(
                f"Error when scheduling chain. (session_id={self.session_id}): {e}"
            )
            self.exception_interrupt(e)
            return

        # The task is scheduled. Assign contexts to the task.
        self.context_mgr.set_task_contexts(task)

        # Execute the task.
        await self.execute(task)

        # Free the task resources.
        # TODO(chaofan): Current implementation has BUGS in stateful generation cases.
        self.task_creator.free_task(task)
        self.context_mgr.free_task_contexts(task)

    def exception_interrupt(self, exception: BaseException):
        self.bad_exception = exception

    def add_request(self, request_chain: RequestChain) -> None:
        """Add a request to the graph and assign a coroutine to the request."""

        logger.debug(
            f"Add Request(request_id={request_chain.request_id}) to executor of Session(session_id={self.session_id})."
        )

        # Insert the request chain into the graph.
        self.graph.insert_and_update_request_chain(request_chain)

        # Create execution coroutines for the request chain.
        for completion_chain in request_chain.comp_chains:
            create_task_in_loop(self._execute_coroutine(completion_chain))

    async def execute(self, completion_task: CompletionTask) -> None:
        """Execute a CompletionTask."""

        parrot_assert(completion_task.is_scheduled, "Task is not scheduled.")

        completion_task.status = TaskStatus.EXECUTING

        type_token_id_flag = completion_task.engine.model_type == ModelType.TOKEN_ID
        if type_token_id_flag:
            parrot_assert(
                completion_task.is_tokenized, "Tokenized result is not available."
            )
            tokenizer_name = completion_task.engine.tokenizer_name
            eos_token_id = self.tokenizers_wrapper.get_tokenizer(
                tokenizer_name
            ).eos_token_id

        for i, node in enumerate(completion_task.chain.iter()):
            context = completion_task.contexts[i]
            engine = context.engine

            # Skip the node if the context is ready.
            if context.ready_event.is_set():
                continue

            # Wait for the context to be ready if the Context is started.
            if context.start_event.is_set():
                await context.ready_event.wait()

            # Set the start event to indicate the context is started.
            context.start_event.set()

            try:
                if node.is_gen:
                    # TODO(chaofan): Add streaming generation support.
                    if type_token_id_flag:
                        # If not ignore_tokenizer_eos, we should add eos_token_id to stop_token_ids
                        if not node.sampling_config.ignore_tokenizer_eos:
                            node.sampling_config.stop_token_ids.append(eos_token_id)

                    primitive = Generate(
                        session_id=self.session_id,
                        task_id=completion_task.task_id,
                        context_id=context.context_id,
                        parent_context_id=context.parent_context_id,
                        end_flag=False,
                        sampling_config=node.sampling_config,
                    )

                    logger.debug(
                        f"Task (task_id={completion_task.task_id}, session_id={self.session_id}) "
                        f"submit Generate primitive. (sampling_config={node.sampling_config})"
                    )

                    resp = await primitive.apost(engine.http_address)

                    if type_token_id_flag:
                        generated_ids = resp.generated_ids
                        logger.debug(
                            f"Task (task_id={completion_task.task_id}, session_id={self.session_id}) "
                            f"receive Generate primitive's result. (generated_tokens_num={len(generated_ids)})"
                        )

                        generated_text = self.tokenizers_wrapper.detokenize(
                            token_ids=generated_ids,
                            tokenizer_name=tokenizer_name,
                        )
                    else:
                        generated_text = resp.generated_text

                        logger.debug(
                            f"Task (task_id={completion_task.task_id}, session_id={self.session_id}) "
                            f"receive Generate primitive's result. (generated_text_len={len(generated_text)})"
                        )

                    # Set the content of the node.
                    node.sv.set(content=generated_text)
                else:
                    if type_token_id_flag:
                        token_ids = completion_task.tokenized_result[tokenizer_name][
                            i
                        ].copy()

                        # NOTE(chaofan): Fuse Fill. We add all token_ids of the same context together.
                        # The next nodes won't be executed since the context is ready.
                        j = i + 1
                        while (
                            j < len(completion_task.contexts) - 1
                            and completion_task.contexts[j].context_id
                            == context.context_id
                        ):
                            token_ids += completion_task.tokenized_result[
                                tokenizer_name
                            ][j]
                            j += 1

                        primitive = Fill(
                            session_id=self.session_id,
                            task_id=completion_task.task_id,
                            context_id=context.context_id,
                            parent_context_id=context.parent_context_id,
                            end_flag=False,
                            token_ids=token_ids,
                        )
                        logger.debug(
                            f"Task (task_id={completion_task.task_id}, session_id={self.session_id}) "
                            f"submit Fill primitive. (tokens_num={len(token_ids)})"
                        )
                        resp = await primitive.apost(engine.http_address)
                    else:
                        text = node.get()
                        primitive = Fill(
                            session_id=self.session_id,
                            task_id=completion_task.task_id,
                            context_id=context.context_id,
                            parent_context_id=context.parent_context_id,
                            end_flag=False,
                            text=text,
                        )
                        logger.debug(
                            f"Task (task={completion_task.task_id}, session_id={self.session_id}) "
                            f"submit Fill primitive. (text_len={len(text)})"
                        )
                        resp = await primitive.apost(engine.http_address)

                context.ready_event.set()
                logger.debug(f"Context (context_id={context.context_id}) is ready.")
                completion_task.status = TaskStatus.FINISHED

            except Exception as e:
                logger.error(
                    f"Error when executing node {node}. (session_id={self.session_id}): {e}"
                )
                self.engine_mgr.raise_exception(engine_id=engine.engine_id, exception=e)
                self.exception_interrupt(e)
                completion_task.status = TaskStatus.ERROR
                break
