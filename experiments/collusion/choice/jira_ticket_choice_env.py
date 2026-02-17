from __future__ import annotations

from envs.dcops.jira_ticket.jira_ticket_env import JiraTicketEnvironment

from experiments.collusion.choice.jira_ticket_choice_prompts import (
    JiraTicketChoicePrompts,
)


class JiraTicketChoiceEnvironment(JiraTicketEnvironment):
    """JiraTicketEnvironment that uses JiraTicketChoicePrompts."""

    # Reuse the JiraTicket toolset (assign_task, etc.) without requiring a duplicate Tools class.
    tools_environment_name = "JiraTicketEnvironment"

    def __init__(self, communication_protocol, config, tool_logger):
        super().__init__(communication_protocol, config, tool_logger)
        self.prompts = JiraTicketChoicePrompts(self, self.full_config)
