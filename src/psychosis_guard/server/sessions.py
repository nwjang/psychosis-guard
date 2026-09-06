"""In-memory, TTL + LRU bounded session store with a per-session lock.

One `PsychosisGuard` per conversation. This store is process-local: run one
replica, use sticky sessions, or use the stateless `/v1/chat/completions`
path (no session id -> trajectory rebuilt from the client-sent history).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from ..guard import PsychosisGuard


@dataclass
class Session:
    id: str
    guard: PsychosisGuard
    created_at: float
    last_seen: float
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionStore:
    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 10_000) -> None:
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str, factory: Callable[[], PsychosisGuard]) -> Session:
        now = time.time()
        with self._lock:
            self._sweep(now)
            s = self._sessions.get(session_id)
            if s is None:
                s = Session(id=session_id, guard=factory(), created_at=now, last_seen=now)
                self._sessions[session_id] = s
                while len(self._sessions) > self.max_sessions:
                    self._sessions.popitem(last=False)  # evict least recently used
            else:
                s.last_seen = now
                self._sessions.move_to_end(session_id)
            return s

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._sweep(time.time())
            s = self._sessions.get(session_id)
            if s is not None:
                s.last_seen = time.time()
                self._sessions.move_to_end(session_id)
            return s

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _sweep(self, now: float) -> None:
        cutoff = now - self.ttl
        stale = [k for k, s in self._sessions.items() if s.last_seen < cutoff]
        for k in stale:
            del self._sessions[k]
