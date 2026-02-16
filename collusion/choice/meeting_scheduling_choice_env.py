from __future__ import annotations

from envs.dcops.meeting_scheduling.meeting_scheduling_env import (
    MeetingSchedulingEnvironment,
)

from experiments.collusion.choice.meeting_scheduling_choice_prompts import (
    MeetingSchedulingChoicePrompts,
)


class MeetingSchedulingChoiceEnvironment(MeetingSchedulingEnvironment):
    """MeetingSchedulingEnvironment that uses MeetingSchedulingChoicePrompts."""

    tools_environment_name = "MeetingSchedulingEnvironment"

    def __init__(self, communication_protocol, config, tool_logger):
        super().__init__(communication_protocol, config, tool_logger)
        self.prompts = MeetingSchedulingChoicePrompts(self, self.full_config)
