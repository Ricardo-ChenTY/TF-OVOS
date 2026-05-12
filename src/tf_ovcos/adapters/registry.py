from __future__ import annotations

from dataclasses import dataclass

from tf_ovcos.adapters.base import MethodAdapter
from tf_ovcos.adapters.debug import CopyGroundTruthAdapter, EmptyMaskAdapter
from tf_ovcos.adapters.planned import (
    GroundingDinoSamAdapter,
    ProxyClipAdapter,
    SamAmgClipAdapter,
    SclipAdapter,
)


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    adapter_cls: type[MethodAdapter]
    runnable: bool
    setup_hint: str = ""


_ADAPTER_CLASSES: tuple[type[MethodAdapter], ...] = (
    CopyGroundTruthAdapter,
    EmptyMaskAdapter,
    GroundingDinoSamAdapter,
    SamAmgClipAdapter,
    SclipAdapter,
    ProxyClipAdapter,
)

ADAPTERS: dict[str, type[MethodAdapter]] = {adapter.name: adapter for adapter in _ADAPTER_CLASSES}


def adapter_info() -> list[AdapterInfo]:
    rows: list[AdapterInfo] = []
    for name in sorted(ADAPTERS):
        adapter_cls = ADAPTERS[name]
        rows.append(
            AdapterInfo(
                name=name,
                adapter_cls=adapter_cls,
                runnable=getattr(adapter_cls, "runnable", True),
                setup_hint=getattr(adapter_cls, "setup_hint", ""),
            )
        )
    return rows
