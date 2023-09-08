from .session import Session
from ..orchestration.controller import Controller


class Dispatcher:
    """Dispatcher will dispatch promises to different engines."""

    def __init__(self, controller: Controller):
        self.controller = controller

    def dispatch(self, session: Session):
        # TODO(chaofan): Model selection, speculative dispatching.
        for engine_name, engine in self.controller.engines_table.items():
            session.engine_name = engine_name
            session.engine = engine
            return
