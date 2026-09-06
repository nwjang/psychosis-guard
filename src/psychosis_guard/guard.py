"""PsychosisGuard — the orchestrator that wires the five rail stages.

Per user turn:

    STAGE 1  INPUT RAIL        pre-response risk estimate on the user turn
    STAGE 2  DIALOG RAIL       Mode A system-prompt selection + chatbot call
    STAGE 3  OUTPUT RAIL       risk signals on the bot reply
    STAGE 4  TRAJECTORY RAIL   cumulative state; slope-aware risk   ★ NEW
    STAGE 5  ACTION RAIL       policy decision + Mode B rewrite

If the INPUT rail already sees HIGH risk (and the condition intervenes), the
pipeline short-circuits: the chatbot is never called and a safe response is
returned instead — the NeMo-style "block/replace before further processing".
"""
from __future__ import annotations

import sys

from .config import PolicyConfig
from .detector import RiskSignalDetector, lexical_signals
from .executor import VALID_MODES, InterventionExecutor
from .interfaces import Chatbot, Judge, Rewriter
from .policy import InterventionPolicyEngine
from .rails import ActionRail, DialogRail, InputRail, OutputRail, TrajectoryRail
from .tracker import TrajectoryStateTracker
from .types import RiskLevel, Role, Turn, TurnRecord

# ADVISER-REVIEW: runtime banner wording (domain disclaimer).
_BANNER = (
    "psychosis-guard: research/educational tool. NOT a medical device, NOT "
    "diagnosis or crisis support. In an emergency contact local emergency "
    "services or a crisis line."
)

# ADVISER-REVIEW: safe replacement reply used on a HIGH-risk short-circuit.
# ADVISER-REVIEW: clinical wording — must be reviewed by the mental-health adviser.
_SAFE_REPLY = (
    "I want to pause here rather than keep going down this path. I'm "
    "concerned about how this is affecting you, and I think it would help to "
    "talk with a mental health professional or someone you trust. If you are "
    "in immediate distress, please contact local emergency services or a "
    "crisis line."
)

_banner_shown = False


def _show_banner_once() -> None:
    global _banner_shown
    if not _banner_shown:
        print(_BANNER, file=sys.stderr)
        _banner_shown = True


class PsychosisGuard:
    """Drop-in conversational proxy: `guard.send(user_text) -> reply`."""

    def __init__(
        self,
        chatbot: Chatbot,
        judge: Judge,
        mode: str = "combined",
        rewriter: Rewriter | None = None,
        policy: PolicyConfig | None = None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(VALID_MODES)}")
        self.mode = mode
        self.tracker = TrajectoryStateTracker()
        policy_engine = InterventionPolicyEngine(policy)
        executor = InterventionExecutor(mode, rewriter)
        self._slope_window = policy_engine.cfg.slope_window

        # the single-turn-filter baseline is the ONLY condition with the
        # Trajectory Rail disabled — the turn-local baseline
        trajectory_enabled = mode != "single-turn-filter"

        self.input_rail = InputRail(policy_engine)
        self.dialog_rail = DialogRail(executor, chatbot)
        self.output_rail = OutputRail(RiskSignalDetector(judge))
        self.trajectory_rail = TrajectoryRail(
            self.tracker, enabled=trajectory_enabled, slope_window=self._slope_window
        )
        self.action_rail = ActionRail(policy_engine, executor)

        # short-circuiting is an intervention, so only intervening modes do it
        self._can_intervene = mode not in ("none", "detect-only")

        self.history: list[Turn] = []
        self.log: list[TurnRecord] = []

    @classmethod
    def from_config(
        cls,
        path: str | None = None,
        *,
        chatbot: Chatbot | None = None,
        judge: Judge | None = None,
        rewriter: Rewriter | None = None,
    ) -> PsychosisGuard:
        """Build a guard from a NeMo-style YAML config (see config.yml).

        Components not supplied (chatbot/judge/rewriter) default to the
        deterministic mocks, so this runs with no API key.
        """
        from .config import load_config
        from .mocks import MockChatbot, MockJudge, MockRewriter

        cfg = load_config(path)
        return cls(
            chatbot=chatbot or MockChatbot(),
            judge=judge or MockJudge(),
            mode=cfg.condition,
            rewriter=rewriter or MockRewriter(),
            policy=cfg.policy,
        )

    def send(self, user_text: str, *, bot_reply: str | None = None) -> str:
        """Run one user turn through Stages 1→5; returns the final reply.

        `bot_reply` is the deployment "check-only" path: the caller's own
        chatbot has already produced a reply and the middleware only scores
        and post-processes it (Mode B). Stage 2 then never calls the wrapped
        chatbot and Mode A cannot apply — the trace records this. The HIGH
        short-circuit still replaces the reply, so the caller must use the
        returned text, not the one it passed in.
        """
        _show_banner_once()
        record = TurnRecord(user_text=user_text)
        self.history.append(Turn(Role.USER, user_text))

        # Stage 1 — input rail, using the slope of the PRIOR trajectory
        # (withheld when the Trajectory Rail is disabled)
        prior_slope = (
            self.tracker.state.delusion_slope(self._slope_window)
            if self.trajectory_rail.enabled
            else 0.0
        )
        pre_level = self.input_rail.run(record, prior_slope)

        if pre_level == RiskLevel.HIGH and self._can_intervene:
            final_reply = self._short_circuit(record)
        else:
            if bot_reply is None:
                # Stage 2 — dialog rail (Mode A prompt + chatbot call)
                self.dialog_rail.run(self.history, record, pre_level)
            else:
                # Stage 2 — externally produced reply; Mode A not applicable
                record.raw_reply = bot_reply
                record.trace(
                    self.dialog_rail.STAGE,
                    f"external reply supplied; mode_a_prompt=n/a (pre_level={pre_level.name})",
                )
            # Stage 3 — output rail (signals on the raw reply)
            self.output_rail.run(self.history, record)
            # Stage 4 — trajectory rail (update state, slope-aware risk)
            slope = self.trajectory_rail.run(record)
            # Stage 5 — action rail (final risk, level, Mode B)
            final_reply = self.action_rail.run(self.history, record, slope)

        self.tracker.state.levels_decided.append(record.level)
        self.history.append(Turn(Role.BOT, final_reply))
        self.log.append(record)
        return final_reply

    def prime(self, history: list[Turn]) -> None:
        """Seed the guard with a prior conversation (stateless deployments).

        OpenAI-style clients resend the whole message list on every request,
        so a fresh guard can rebuild the trajectory from it: each earlier
        user/assistant pair is scored with the API-free lexical rubric only
        (no judge calls) and folded into the tracker. Call once, before the
        first `send`. A trailing user turn without a reply is ignored — pass
        it to `send` instead.
        """
        if self.history or self.log:
            raise RuntimeError("prime() must be called on a fresh guard")
        pending_user: str | None = None
        for turn in history:
            if turn.role == Role.USER:
                pending_user = turn.text
                continue
            if pending_user is None:
                # assistant turn with no preceding user turn: keep for context only
                self.history.append(Turn(Role.BOT, turn.text))
                continue
            signals = lexical_signals(pending_user, turn.text)
            self.tracker.update(
                pending_user,
                delusion_density=signals.delusion_density,
                conviction=signals.conviction,
                bot_validated=bool(signals.bot_validated),
                isolation=bool(signals.isolation),
            )
            self.tracker.state.levels_decided.append(RiskLevel.NONE)
            self.history.append(Turn(Role.USER, pending_user))
            self.history.append(Turn(Role.BOT, turn.text))
            pending_user = None

    def summary(self) -> dict:
        """JSON-serializable view of the trajectory state (for APIs/logs)."""
        st = self.tracker.state
        last = self.log[-1] if self.log else None
        return {
            "mode": self.mode,
            "turn_count": st.turn_count,
            "delusion_slope": st.delusion_slope(),
            "recent_slope": st.delusion_slope(self._slope_window),
            "bot_validation_count": st.bot_validation_count,
            "isolation_mentions": st.isolation_mentions,
            "levels_decided": [lvl.name for lvl in st.levels_decided],
            "last_level": last.level.name if last else None,
            "last_risk": last.composite_risk if last else None,
        }

    def _short_circuit(self, record: TurnRecord) -> str:
        """HIGH at Stage 1: block the chatbot call, substitute the safe reply.

        The remaining stages still run in decide/measure form so the
        TurnRecord and trajectory state stay complete for analysis.
        """
        record.trace(
            "stage2:dialog", "SHORT-CIRCUIT: chatbot call blocked (pre_level=HIGH)"
        )
        record.raw_reply = ""
        record.final_reply = _SAFE_REPLY
        # score the safe reply so user-side signals & bot-side DVs stay logged
        record.signals = self.output_rail.detector.detect(
            self.history, record.user_text, _SAFE_REPLY
        )
        record.trace("stage3:output", "signals scored on safe reply")
        slope = self.trajectory_rail.run(record)
        self.action_rail.decide_only(record, slope)
        return _SAFE_REPLY
