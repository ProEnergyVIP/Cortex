from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

FailureStrategy = Literal["raise", "fallback"]


@dataclass(frozen=True)
class StepPolicy:
    max_retries: int = 0
    failure_strategy: FailureStrategy = "raise"
    fallback_step: Optional[str] = None
    timeout_seconds: Optional[float] = None

    def __post_init__(self):
        if self.max_retries < 0:
            raise ValueError("StepPolicy.max_retries must be >= 0")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("StepPolicy.timeout_seconds must be > 0 when provided")
        if self.failure_strategy == "fallback" and not self.fallback_step:
            raise ValueError("StepPolicy with failure_strategy='fallback' requires fallback_step")
        if self.failure_strategy != "fallback" and self.fallback_step is not None:
            raise ValueError("fallback_step can only be set when failure_strategy='fallback'")
