"""Common interface for stream-based OOD methods operating on cached CLIP features."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class StepOutput:
    pred: torch.Tensor           # (B,) predicted ID class
    score: torch.Tensor          # (B,) ID-ness score (higher = more ID)
    info: Dict[str, Any] = field(default_factory=dict)  # optional diagnostics


class StreamMethod:
    """A test-time method sees one batch of L2-normalised image features at a time.

    ``reset()`` is called at the start of every evaluation stream (the official
    evaluator resets memory for every ID/OOD pair). ``step()`` must return the
    prediction and score for the batch, and may update internal memory.
    """

    name = "base"

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    def step(self, feats: torch.Tensor) -> StepOutput:  # pragma: no cover - interface
        raise NotImplementedError

    # Methods that do not adapt can be scored in one big batch.
    stateless = False

    def config_dict(self) -> Dict[str, Any]:
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            return {}
        try:
            from dataclasses import asdict

            return asdict(cfg)
        except TypeError:
            return dict(vars(cfg))


def to_device(x: Optional[torch.Tensor], device) -> Optional[torch.Tensor]:
    return None if x is None else x.to(device)
