from typing import TypedDict, Optional, List, Dict
from app.core.schema import ResumeFacts, JDRequirements, ResumeCritique


class LoopState(TypedDict):
    jd: JDRequirements
    original_facts: ResumeFacts      # untouched source of truth, never modified
    facts: ResumeFacts                 # current working copy, updated each iteration
    score_breakdown: Optional[Dict]
    critique: Optional[ResumeCritique]
    iteration: int
    max_iterations: int
    target_score: float
    score_history: List[float]         # tracks score after every iteration, used to detect plateau
    status: str                        # "in_progress" | "success" | "plateaued" | "max_iterations_reached"