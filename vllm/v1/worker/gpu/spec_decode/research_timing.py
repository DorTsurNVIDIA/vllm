# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional research diagnostics, resolved only at completed-request boundaries."""

import torch


class ComponentTiming:
    """Record GPU events without synchronizing the per-round hot path.

    Use in separate diagnostic jobs. Event allocation and recording add overhead;
    instrumented wall time must not enter the main latency frontier.
    """

    def __init__(self):
        self.pending: dict[str, torch.cuda.Event] = {}
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def start(self, component: str) -> None:
        if component in self.pending:
            raise RuntimeError("Unclosed component timing interval")
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.pending[component] = event

    def stop(self, component: str) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.events.append((component, self.pending.pop(component), end))

    def finish(self) -> dict[str, dict[str, float]]:
        if self.pending:
            raise RuntimeError("A component interval did not finish")
        if self.events:
            self.events[-1][2].synchronize()
        result: dict[str, dict[str, float]] = {}
        for component, start, end in self.events:
            row = result.setdefault(component, {"milliseconds": 0.0, "calls": 0.0})
            row["milliseconds"] += start.elapsed_time(end)
            row["calls"] += 1
        if "verify" in result:
            result["verify_excluding_solve"] = {
                "milliseconds": max(
                    0.0,
                    result["verify"]["milliseconds"]
                    - result.get("solve", {}).get("milliseconds", 0.0),
                ),
                "calls": result["verify"]["calls"],
            }
        return result
