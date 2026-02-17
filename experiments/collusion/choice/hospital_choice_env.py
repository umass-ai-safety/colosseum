from __future__ import annotations

from envs.dcops.hospital.hospital_env import HospitalEnvironment

from experiments.collusion.choice.hospital_choice_prompts import HospitalChoicePrompts


class HospitalChoiceEnvironment(HospitalEnvironment):
    """HospitalEnvironment that uses HospitalChoicePrompts."""

    tools_environment_name = "HospitalEnvironment"

    def __init__(self, communication_protocol, config, tool_logger):
        super().__init__(communication_protocol, config, tool_logger)
        self.prompts = HospitalChoicePrompts(self, self.full_config)

