from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Set

from terrarium.agents.base import BaseAgent


@dataclass(frozen=True, slots=True)
class AdversarySpec:
    """Configuration controlling adversarial behavior."""

    adversary_names: Set[str]
    misinformation_claim: str
    force_seed: bool = True


class NetworkInfluenceMeetingAgent(BaseAgent):
    """BaseAgent subclass with optional adversary metadata."""

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
