# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List, Optional, Tuple

from parrot.exceptions import ParrotCoreUserError, parrot_assert
from parrot.utils import RecyclePool, get_logger, time_counter_in_nanoseconds

from .session.session import Session, SessionStatus


logger = get_logger("SessionManager")


class SessionManager:
    """
    Manage all sessions connected to the cluster.
    """

    def __init__(self, **session_create_kwargs) -> None:
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
        session.free_session()
        self.session_id_pool.free(session_id)

        logger.debug(f"Session {session_id} is removed.")

    # ---------- Methods for Core ----------

    def register_session(self) -> int:
        """Create a new session.

        Returns:
            int: The session ID.
        """

        session_id = self.session_id_pool.allocate()
        session = Session(session_id=session_id, **self.session_create_kwargs)

        self.sessions[session_id] = session
        self.session_last_access_time[session_id] = time_counter_in_nanoseconds()

        logger.debug(f"Session (id={session_id}) registered.")
        return session_id

    def get_session(self, session_id: int) -> Session:
        """Get the session by session ID.

        Args:
            session_id (int): The session ID.

        Returns:
            Optional[Session]: The session.
        """

        parrot_assert(
            session_id in self.sessions,
            f"Session {session_id} not found.",
        )
        return self.sessions[session_id]

    def session_access_update(self, session_id: int) -> None:
        """Update the last access time of the session.

        Args:
            session_id (int): The session ID.
        """

        parrot_assert(
            session_id in self.sessions,
            f"Session {session_id} not found.",
        )
        self.session_last_access_time[session_id] = time_counter_in_nanoseconds()

    def check_session_status(self, session_id: int) -> None:
        """Check the status of the session.

        Args:
            session_id (int): The session ID.
        """

        if session_id not in self.sessions:
            raise ParrotCoreUserError(RuntimeError(f"Session {session_id} not found."))

        session = self.sessions[session_id]
        if session.status != SessionStatus.RUNNING:
            raise ParrotCoreUserError(
                RuntimeError(f"Session {session_id} is not valid.")
            )

    async def check_running_sessions(self) -> None:
        """1. If the session is expired, mark it as DEAD.
        2. If the executor of the session raises an exception, mark it as BAD.
        """

        current_time = time_counter_in_nanoseconds()
        for session_id, last_access_time in self.session_last_access_time.items():
            session = self.sessions[session_id]

            if not session.is_running:
                continue

            if current_time - last_access_time > session.life_span * 1_000_000_000:
                session.status = SessionStatus.DEAD
                logger.debug(f"Session {session_id} is expired.")
            elif session.executor.bad_exception is not None:
                session.status = SessionStatus.BAD
                logger.debug(f"Session {session_id} is bad.")

    async def sweep_not_running_sessions(self) -> None:
        """Sweep the dead/bad sessions."""

        for session_id, session in self.sessions.items():
            if not session.is_running:
                self._remove_session(session_id)
