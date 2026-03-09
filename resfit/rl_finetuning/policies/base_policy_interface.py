from __future__ import annotations

from typing import Any, Protocol


class BaseChunkPolicy(Protocol):
    def get_modality_config(self) -> Any:
        ...

    def infer_action_chunk(self, obs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        ...
