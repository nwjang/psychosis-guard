"""psychosis-guard — trajectory-aware safety middleware for chatbots.

Mitigates "AI psychosis": the gradual amplification of a user's delusional
beliefs across a conversation, driven by chatbot sycophancy. Wraps any chatbot
behind a single interface and adds a Trajectory Rail that tracks cumulative
risk over the whole conversation — not just the current turn.

DISCLAIMER: Research/educational tool. NOT a medical device, NOT diagnosis or
crisis support. In an emergency contact local emergency services or a crisis
line.
"""

__version__ = "0.2.0"

from .config import GuardConfig, PolicyConfig, load_config
from .guard import PsychosisGuard
from .types import RiskLevel, Role, Signals, Turn, TurnRecord

__all__ = [
    "GuardConfig",
    "PolicyConfig",
    "PsychosisGuard",
    "RiskLevel",
    "Role",
    "Signals",
    "Turn",
    "TurnRecord",
    "__version__",
    "load_config",
]
