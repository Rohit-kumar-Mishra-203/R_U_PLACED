from langgraph.graph import StateGraph, END
from app.graph.state import LoopState
from app.graph.nodes import score_node, critique_node, edit_node, route_after_score


def build_graph():
    graph = StateGraph(LoopState)

    graph.add_node("score", score_node)
    graph.add_node("critique", critique_node)
    graph.add_node("edit", edit_node)

    graph.set_entry_point("score")

    graph.add_conditional_edges(
        "score",
        route_after_score,
        {
            "stop": END,
            "continue": "critique",
        },
    )

    graph.add_edge("critique", "edit")
    graph.add_edge("edit", "score")

    return graph.compile()


def run_optimization_loop(
    jd,
    base_facts,
    target_score: float = 93.0,
    max_iterations: int = 6,
):
    app = build_graph()

    initial_state: LoopState = {
        "jd": jd,
        "original_facts": base_facts,
        "facts": base_facts,
        "score_breakdown": None,
        "critique": None,
        "iteration": 0,
        "max_iterations": max_iterations,
        "target_score": target_score,
        "score_history": [],
        "status": "in_progress",
    }

    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    from app.core.schema import ResumeFacts, JDRequirements

    with open("data/resume_facts.json", "r", encoding="utf-8") as f:
        facts = ResumeFacts.model_validate_json(f.read())

    sample_jd = JDRequirements(
        job_title="Machine Learning Engineer",
        company=None,
        seniority_level="Mid-Level",
        must_have_skills=["Python", "PyTorch", "Transformer architectures", "LLM fine-tuning"],
        nice_to_have_skills=["LangChain/LangGraph", "RAG systems"],
        responsibilities=[
            "Build and deploy production NLP pipelines",
            "Collaborate with product team",
            "Optimize model performance",
        ],
        tools_and_tech=["Python", "PyTorch", "LangChain", "LangGraph"],
        raw_text="",
    )

    result = run_optimization_loop(sample_jd, facts)

    print(f"\nFinal status: {result['status']}")
    print(f"Iterations run: {result['iteration']}")
    print(f"Score history: {result['score_history']}")

    if result["status"] == "plateaued" and result["critique"]:
        genuine_gaps = [item for item in result["critique"].items if item.is_genuine_gap]
        if genuine_gaps:
            print("\n--- Honest gaps that couldn't be closed by rephrasing ---")
            for gap in genuine_gaps:
                print(f"- {gap.description}")