from __future__ import annotations

from envs.dcops.smart_grid.smart_grid_env import SmartGridEnvironment

from experiments.collusion.choice.smart_grid_choice_prompts import SmartGridChoicePrompts


class SmartGridChoiceEnvironment(SmartGridEnvironment):
    """SmartGridEnvironment that uses SmartGridChoicePrompts."""

    tools_environment_name = "SmartGridEnvironment"

    def __init__(self, communication_protocol, config, tool_logger):
        super().__init__(communication_protocol, config, tool_logger)
        self.prompts = SmartGridChoicePrompts(self, self.full_config)
