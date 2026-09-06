"""The five rail stages of the pipeline.

Stages 1–3 and 5 mirror NeMo Guardrails' input/dialog/output/action rails;
Stage 4 (Trajectory Rail) is the novel, cumulative-state stage.
"""

from .action_rail import ActionRail
from .dialog_rail import DialogRail
from .input_rail import InputRail
from .output_rail import OutputRail
from .trajectory_rail import TrajectoryRail

__all__ = ["ActionRail", "DialogRail", "InputRail", "OutputRail", "TrajectoryRail"]
