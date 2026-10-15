import os
import copy
from typing import cast, List, Dict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.schema import ResumeFacts, ResumeCritique, CritiqueItem, EditorOutput

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

structured_llm = llm.with_structured_output(EditorOutput)

EDITOR_PROMPT = """You are revising specific resume bullet points based on feedback,
to better match a job description - WITHOUT ever inventing new facts.

CRITICAL RULES - follow these strictly:
1. You may ONLY revise the exact bullets listed below by their id. Do not
   touch, mention, or create any other bullet.
2. You may ONLY rephrase, reframe, or re-emphasize what is ALREADY stated in
   that bullet's original text. You may pull in terminology/emphasis implied
   by the suggestion, but you must NEVER add a new tool, technology, metric,
   number, or claim that was not already present in the original bullet text.
3. If the suggestion implies adding something not actually in the original
   bullet (a new metric, a new tool never mentioned), do NOT add it - instead
   revise as close as honestly possible without fabricating, and note the
   limitation in your justification.
4. Keep each revised bullet a similar length to the original - this is a
   rephrase, not an expansion into a paragraph.
5. justification must briefly state which part of the ORIGINAL bullet
   grounds your revision - this is a factual traceability check, not
   marketing language.

Bullets to revise (id, original text, and the specific suggestion for it):
{bullets_to_revise}
"""


def _format_bullets_for_editing(facts: ResumeFacts, items: List[CritiqueItem]) -> str:
    id_to_text: Dict[str, str] = {}
    for exp in facts.experience:
        for b in exp.bullets:
            id_to_text[b.id] = b.text
    for proj in facts.projects:
        for b in proj.bullets:
            id_to_text[b.id] = b.text

    lines = []
    for item in items:
        if item.target_bullet_id and item.target_bullet_id in id_to_text:
            original = id_to_text[item.target_bullet_id]
            lines.append(
                f"- id: {item.target_bullet_id}\n"
                f"  original: \"{original}\"\n"
                f"  suggestion: \"{item.suggestion}\""
            )
    return "\n".join(lines)


def generate_revisions(facts: ResumeFacts, critique: ResumeCritique) -> EditorOutput:
    actionable_items = [
        item for item in critique.items
        if not item.is_genuine_gap and item.target_bullet_id
    ]

    if not actionable_items:
        return EditorOutput(revisions=[])

    bullets_text = _format_bullets_for_editing(facts, actionable_items)
    prompt = EDITOR_PROMPT.format(bullets_to_revise=bullets_text)
    result = cast(EditorOutput, structured_llm.invoke(prompt))
    return result


def apply_revisions(facts: ResumeFacts, editor_output: EditorOutput) -> ResumeFacts:
    """
    Returns a NEW ResumeFacts object with only the targeted bullets' text
    updated. The original facts object (and file on disk) is never mutated.
    """
    updated = copy.deepcopy(facts)
    revision_map = {r.bullet_id: r.revised_text for r in editor_output.revisions}

    for exp in updated.experience:
        for b in exp.bullets:
            if b.id in revision_map:
                b.text = revision_map[b.id]

    for proj in updated.projects:
        for b in proj.bullets:
            if b.id in revision_map:
                b.text = revision_map[b.id]

    return updated


if __name__ == "__main__":
    from app.core.scorer import score_resume
    from app.core.critic import generate_critique
    from app.core.schema import JDRequirements

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

    score_result = score_resume(sample_jd, facts)
    critique = generate_critique(sample_jd, facts, score_result)
    editor_output = generate_revisions(facts, critique)

    print("--- Revisions ---")
    print(editor_output.model_dump_json(indent=2))

    updated_facts = apply_revisions(facts, editor_output)

    print("\n--- Before / After ---")
    revision_map = {r.bullet_id: r.revised_text for r in editor_output.revisions}
    for exp in facts.experience:
        for b in exp.bullets:
            if b.id in revision_map:
                print(f"\n[{b.id}] BEFORE: {b.text}")
                print(f"[{b.id}] AFTER:  {revision_map[b.id]}")
    for proj in facts.projects:
        for b in proj.bullets:
            if b.id in revision_map:
                print(f"\n[{b.id}] BEFORE: {b.text}")
                print(f"[{b.id}] AFTER:  {revision_map[b.id]}")

    new_score = score_resume(sample_jd, updated_facts)
    print(f"\nOriginal score: {score_result['overall_score']}")
    print(f"New score:      {new_score['overall_score']}")