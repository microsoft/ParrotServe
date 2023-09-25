from parrot.orchestration.controller import Controller

from .session import Session


class Dispatcher:
    """Dispatcher will dispatch promises to different engines."""

    def __init__(self, controller: Controller):
        self.controller = controller

    def dispatch(self, session: Session):
        # Must be dispatched to the same engine with the shared context.
        if session.promise.shared_context_handler is not None:
            session.engine = (
                session.promise.shared_context_handler.shared_context.engine
            )
            session.engine_name = session.engine.name
            return

        # TODO(chaofan): Model selection, speculative dispatching.
        for engine_name, engine in self.controller.engines_table.items():
            session.engine_name = engine_name
            session.engine = engine
            return
