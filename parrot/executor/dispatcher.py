from .session import Session
from ..orchestration.controller import parrot_global_ctrl


class Dispatcher:
    """Dispatcher will dispatch promises to different engines."""

    def dispatch(self, session: Session):
        # TODO(chaofan): Model selection, speculative dispatching.

        session.engine_name = parrot_global_ctrl.engines_table.values[0]
