"""Pure full-schedule orchestration; optimizer work requires supplied callbacks.

No models/data are created here. This is not a standalone authorized training
runner. Primary is fixed last; source-selected state remains diagnostic only.
"""
from __future__ import annotations

import math
from dataclasses import asdict

from mocd_full_matching import FullSchedule


class ScheduleControl:
    def __init__(self,fingerprint,schedule=None):
        if not fingerprint:raise ValueError("bound data/code/profile fingerprint required")
        self.fingerprint=fingerprint;self.schedule=schedule or FullSchedule()
        self.completed=0;self.actual_updates=0
        self.best_step=None;self.best_score=None
        self.finished=False
        self.failed_step=None

    def run(self,attempt,refresh,transition,source_validate,stop_after=None):
        if self.finished:raise RuntimeError("finished cell cannot silently run again")
        if self.failed_step is not None:raise RuntimeError("failed callback may have changed model; explicit joint checkpoint recovery required")
        limit=self.schedule.total_steps if stop_after is None else stop_after
        if type(limit) is not int or not self.completed<limit<=self.schedule.total_steps:
            raise ValueError("invalid continuation bound")
        for step in range(self.completed+1,limit+1):
            event=self.schedule.event(step)
            try:
                if event["start_B"]:transition(step,event)
                actual=attempt(step,event)
                if type(actual) is not bool:raise ValueError("attempt callback must report actual optimizer update")
                if event["refresh"]:refresh(step,event)
                if event["source_val_diagnostic"]:
                    result=source_validate(step,event)
                    if result.get("target_used_for_selection") is not False:
                        raise ValueError("target cannot choose checkpoint")
                    score=result["source_score"]
                    if not isinstance(score,(float,int)) or not math.isfinite(score) or not 0<=score<=1:
                        raise ValueError("invalid source score")
                    if self.best_score is None or score>self.best_score:
                        self.best_step=step;self.best_score=float(score)
            except Exception:
                self.failed_step=step
                raise
            # A callback failure leaves the cursor before this step. Caller must
            # stop and retain its possibly advanced model; never retry blindly.
            self.completed=step;self.actual_updates+=int(actual)
            self.finished=event["primary_final"]

    def state_dict(self):
        return {"schema":"MOCD_FULL_SCHEDULE_CURSOR_V1","fingerprint":self.fingerprint,
                "schedule":asdict(self.schedule),"completed":self.completed,"actual_updates":self.actual_updates,
                "best_step":self.best_step,"best_score":self.best_score,"finished":self.finished,
                "failed_step":self.failed_step,
                "primary_policy":"fixed_last","target_used_for_selection":False}

    def load_state_dict(self,state):
        if set(state)!=set(self.state_dict()) or state["schema"]!="MOCD_FULL_SCHEDULE_CURSOR_V1" or state["fingerprint"]!=self.fingerprint or state["schedule"]!=asdict(self.schedule):
            raise ValueError("cursor code/data/schedule identity mismatch")
        if state["primary_policy"]!="fixed_last" or state["target_used_for_selection"] is not False:
            raise ValueError("primary/selection policy mismatch")
        completed,updates=state["completed"],state["actual_updates"]
        if type(completed) is not int or type(updates) is not int or not 0<=updates<=completed<=self.schedule.total_steps:
            raise ValueError("cursor update counters invalid")
        if type(state["finished"]) is not bool or state["finished"]!=(completed==self.schedule.total_steps):
            raise ValueError("cursor final flag mismatch")
        failed=state["failed_step"]
        if failed is not None and (type(failed) is not int or failed!=completed+1 or failed>self.schedule.total_steps):
            raise ValueError("failed callback position mismatch")
        best=state["best_step"];score=state["best_score"]
        if (best is None)!=(score is None):raise ValueError("incomplete source selection state")
        if best is not None:
            if type(best) is not int or not self.schedule.A_steps<best<=completed or not self.schedule.event(best)["source_val_diagnostic"] or not math.isfinite(score) or not 0<=score<=1:
                raise ValueError("source checkpoint cursor invalid")
        self.completed=completed;self.actual_updates=updates
        self.best_step=best;self.best_score=score;self.finished=state["finished"]
        self.failed_step=failed

    def primary_ready(self):
        # Per-cell only: a separate full-panel freeze barrier is still required.
        return self.finished and self.completed==self.schedule.total_steps
