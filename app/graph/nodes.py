from app.graph.state import LoopState
from app.core.scorer import score_resume
from app.core.critic import generate_critique
from app.core.editor import generate_revisions, apply_revisions

# If score improves by less than this between iterations, we consider it
# a plateau - further looping is unlikely to help and risks the editor
# grasping for smaller and smaller rephrasing gains.
PLATEAU_THRESHOLD = 0.5


def score_node(state: LoopState) -> LoopState:
    """Scores the current working resume against the JD."""
    breakdown = score_resume(state["jd"], state["facts"])
    state["score_breakdown"] = breakdown
    state["score_history"] = state["score_history"] + [breakdown["overall_score"]]
    return state


def critique_node(state: LoopState) -> LoopState:
    """Generates specific, honest critique based on the current score."""
    assert state["score_breakdown"] is not None
    critique = generate_critique(state["jd"], state["facts"], state["score_breakdown"])
    state["critique"] = critique
    return state


def edit_node(state: LoopState) -> LoopState:
    """
    Applies critique-driven revisions to the working resume copy.
    The original_facts object is never touched - only the working copy
    advances each iteration.
    """
    assert state["critique"] is not None
    editor_output = generate_revisions(state["facts"], state["critique"])
    state["facts"] = apply_revisions(state["facts"], editor_output)
    state["iteration"] = state["iteration"] + 1
    return state


def route_after_score(state: LoopState) -> str:
    """
    Decides what happens after each scoring step:
    - "stop" with status "success": target score reached
    - "stop" with status "plateaued": score isn't meaningfully improving anymore
    - "stop" with status "max_iterations_reached": hit the iteration cap
    - "continue": keep looping
    """
    current_score = state["score_history"][-1]

    if current_score >= state["target_score"]:
        state["status"] = "success"
        return "stop"

    if state["iteration"] >= state["max_iterations"]:
        state["status"] = "max_iterations_reached"
        return "stop"

    # Plateau check: only meaningful after at least 2 scored iterations
    if len(state["score_history"]) >= 2:
        improvement = state["score_history"][-1] - state["score_history"][-2]
        if improvement < PLATEAU_THRESHOLD:
            state["status"] = "plateaued"
            return "stop"

    state["status"] = "in_progress"
    return "continue"