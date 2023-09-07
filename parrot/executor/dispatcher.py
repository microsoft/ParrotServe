from .session import Session
from ..orchestration.controller import Controller


class Dispatcher:
    """Dispatcher will dispatch promises to different engines."""

    def __init__(self, controller: Controller):
        self.controller = controller

    def dispatch(self, session: Session):
        # TODO(chaofan): Model selection, speculative dispatching.

        session.engine_name = self.controller.engines_table.values[0]
