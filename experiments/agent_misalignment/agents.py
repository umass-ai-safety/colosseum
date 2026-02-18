from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Set

from terrarium.agents.base import BaseAgent


@dataclass(frozen=True, slots=True)
class AdversarySpec:
    """Configuration for Misalignment/DoS Adversary."""
    adversary_names: Set[str]
    attack_description: str
    strategy_type: str


class MisalignmentHospitalAgent(BaseAgent):
    """
    Agent capable of executing Strategic Resource Depletion & Front-Running.
    """

    def __init__(
        self,
        *args: Any,
        adversary_spec: Optional[AdversarySpec] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.adversary_spec = adversary_spec

    @property
    def is_adversary(self) -> bool:
        spec = self.adversary_spec
        return bool(spec and self.name in spec.adversary_names)
