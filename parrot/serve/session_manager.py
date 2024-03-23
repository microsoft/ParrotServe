# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List, Optional, Tuple

from parrot.utils import RecyclePool, get_logger, time_counter_in_nanoseconds

from .session.session import Session, SessionStatus


logger = get_logger("SessionManager")


class SessionManager:
    """
    Manage all sessions connected to the cluster.
    """

    def __init__(self, session_create_kwargs: Dict) -> None:
        # ---------- Session Managing ----------
        # session_id -> session
        self.sessions: Dict[int, Session] = {}
        self.session_id_pool: RecyclePool = ()

        # session_id -> last_access_time (nanoseconds)
        self.session_last_access_time: Dict[int, int] = {}

        # ---------- Arguments for Creating Session ----------
        self.session_create_kwargs = session_create_kwargs

    def _remove_session(self, session_id: int) -> None:
        session = self.sessions.pop(session_id)
        self.session_last_access_time.pop(session_id)
        session.free_process()
        self.session_id_pool.free(session_id)

        logger.debug(f"Session {session_id} is removed.")

    # ---------- Methods for Executor ----------

    def raise_exception(self, session_id: int, exception: Exception) -> None:
        """Raise an exception in the session.

        Args:
            session_id (int): The session ID.
            exception (Exception): The exception to be raised.
        """

        session = self.sessions[session_id]
        session.mark_bad(exception=exception)

    # ---------- Methods for Core ----------

    async def register_session(self) -> int:
        """Create a new session.

        Returns:
            int: The session ID.
        """

        session_id = self.session_id_pool.allocate()
        session = Session(session_id=session_id, **self.session_create_kwargs)

        self.sessions[session_id] = session
        self.session_last_access_time[session_id] = time_counter_in_nanoseconds()

        logger.debug(f"Session {session_id} registered.")
        return session_id

    async def session_access_update(self, session_id: int) -> None:
        """Update the last access time of the session.

        Args:
            session_id (int): The session ID.
        """

        self.session_last_access_time[session_id] = time_counter_in_nanoseconds()

    async def update_expired_sessions(self) -> None:
        """If the session is expired, update the session status."""

        current_time = time_counter_in_nanoseconds()
        for session_id, last_access_time in self.session_last_access_time.items():
            session = self.sessions[session_id]
            if current_time - last_access_time > session.life_span * 1_000_000_000:
                session.status = SessionStatus.DEAD
                logger.debug(f"Session {session_id} is expired.")

    async def sweep_not_running_sessions(self) -> None:
        """Sweep the dead/bad sessions."""

        for session_id, session in self.sessions.items():
            if session.status != SessionStatus.RUNNING:
                self._remove_session(session_id)
