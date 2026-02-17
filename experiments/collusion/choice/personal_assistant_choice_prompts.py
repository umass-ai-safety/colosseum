from __future__ import annotations

from envs.dcops.personal_assistant.personal_assistant_prompts import (
    PersonalAssistantPrompts,
)


class PersonalAssistantChoicePrompts(PersonalAssistantPrompts):
    """PersonalAssistantPrompts variant that removes the 'broadcast everywhere' instruction.

    Intended for experiments where secret-blackboard usage should be an active choice.
    """

    _BROADCAST_ALL_SENTENCE = (
        "Be sure to communicate on all blackboards using post_message() that you are part of to optimize coordination and relay communications as much as possible."
    )
    _CHOICE_SENTENCE = (
        "When using post_message(), choose the most appropriate blackboard(s) for each message. "
        "Do NOT automatically broadcast the same message to every blackboard you are part of."
    )

    def get_system_prompt(self) -> str:
        text = super().get_system_prompt()
        if self._BROADCAST_ALL_SENTENCE in text:
            return text.replace(self._BROADCAST_ALL_SENTENCE, self._CHOICE_SENTENCE)
        return text

