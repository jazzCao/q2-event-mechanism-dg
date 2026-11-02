"""Audit instrumentation over the CUDA-qualified engine; objective unchanged."""
from __future__ import annotations

import copy
import math

from mocd_full_engine import FullModelEngine


def validate_source_curve(packet, schedule):
    curve = packet["sampler"]["source_scores"]
    completed = packet["progress"]["cursor"]["completed"]
    if type(completed) is not int or not 0 <= completed <= schedule.total_steps:
        raise ValueError("source curve cursor")
    expected_steps = [step for step in range(1, completed + 1) if schedule.event(step)["source_val_diagnostic"]]
    if not isinstance(curve, list) or [r["step"] for r in curve] != expected_steps:
        raise ValueError("complete source diagnostic curve required")
    for row in curve:
        if set(row) != {"step", "source_score", "target_used_for_selection"} or type(row["step"]) is not int or row["target_used_for_selection"] is not False:
            raise ValueError("source curve identity/target flag")
        score = row["source_score"]
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("source curve score")
    best = min(curve, key=lambda r: (-r["source_score"], r["step"])) if curve else None
    expected_best = (-1, -1.) if best is None else (best["step"], best["source_score"])
    if (packet["selection"]["best_step"], packet["selection"]["best_score"]) != expected_best:
        raise ValueError("earliest source selection not supported by full curve")
    return copy.deepcopy(curve)


class RecordedFullSourceEngine(FullModelEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_scores = []

    def source_validate(self, step, event):
        result = super().source_validate(step, event)
        self.source_scores.append({"step": step, **result})
        return result

    def packet(self):
        packet = super().packet()
        packet["sampler"]["source_scores"] = copy.deepcopy(self.source_scores)
        return packet

    def restore(self, packet):
        curve = validate_source_curve(packet, self.controller.schedule)
        super().restore(packet)
        self.source_scores = curve
