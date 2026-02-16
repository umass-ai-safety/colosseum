from __future__ import annotations

from envs.dcops.personal_assistant.personal_assistant_env import (
    PersonalAssistantEnvironment,
)

from experiments.collusion.choice.personal_assistant_choice_prompts import (
    PersonalAssistantChoicePrompts,
)


class PersonalAssistantChoiceEnvironment(PersonalAssistantEnvironment):
    """PersonalAssistantEnvironment that uses PersonalAssistantChoicePrompts."""

    tools_environment_name = "PersonalAssistantEnvironment"

    def __init__(self, communication_protocol, config, tool_logger):
        super().__init__(communication_protocol, config, tool_logger)
        self.prompts = PersonalAssistantChoicePrompts(self, self.full_config)
