import os
from typing import cast
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from app.core.schema import ResumeFacts, JDRequirements, ResumeCritique
from pydantic import SecretStr


load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=cast(SecretStr, os.getenv("GROQ_API_KEY")),
)

structured_llm = llm.with_structured_output(ResumeCritique, method="json_mode")

CRITIQUE_PROMPT = """You are a career coach reviewing how well a resume matches a job description,
using a scoring breakdown that has already been computed.

Your job is to produce SPECIFIC, ACTIONABLE feedback - not vague advice like
"improve keywords." Every item must be something a resume editor could act on.

CRITICAL RULES - follow these strictly:
1. You may ONLY reference bullets by their real id, taken from the resume data
   below. Never invent a bullet or claim content that isn't there.
2. target_bullet_id MUST be the exact same bullet you discuss in description.
   Before finalizing each item, re-check that the id in target_bullet_id
   matches the bullet your description is actually talking about - these
   must never disagree.
3. gap_type must be exactly one of: "missing_keyword", "weak_semantic_match",
   or "phrasing" - no other values or variations are allowed.
4. When a missing skill or weak match reflects a GENUINE gap - meaning the
   candidate's real experience does not honestly support it, even with
   better phrasing - set is_genuine_gap=True and do NOT suggest a rewrite
   that would misrepresent their experience. Say so plainly instead.
5. When a weak match exists because the right experience is there but
   under-emphasized or poorly phrased, set is_genuine_gap=False and give a
   specific suggestion referencing the exact bullet id to strengthen.
6. Never suggest adding a skill, tool, or claim that isn't already present
   somewhere in the candidate's real resume data below.
7. Prioritize must-have skill gaps and the weakest semantic matches first.

Job description requirements:
{jd_requirements}

Score breakdown (already computed):
{score_breakdown}

Candidate's real resume data (bullets with their real ids - you may ONLY
reference these ids, never invent new ones):
{resume_bullets}
"""


def _format_resume_bullets(facts: ResumeFacts) -> str:
    lines = []
    for exp in facts.experience:
        lines.append(f"\n{exp.company} - {exp.role}:")
        for b in exp.bullets:
            lines.append(f"  [{b.id}] {b.text}")
    for proj in facts.projects:
        lines.append(f"\nProject: {proj.name}:")
        for b in proj.bullets:
            lines.append(f"  [{b.id}] {b.text}")
    return "\n".join(lines)


def generate_critique(jd: JDRequirements, facts: ResumeFacts, score_breakdown: dict) -> ResumeCritique:
    prompt = CRITIQUE_PROMPT.format(
        jd_requirements=jd.model_dump_json(indent=2),
        score_breakdown=score_breakdown,
        resume_bullets=_format_resume_bullets(facts),
    )
    result = cast(ResumeCritique, structured_llm.invoke(prompt))
    return result


if __name__ == "__main__":
    import json
    from app.core.scorer import score_resume

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

    print(critique.model_dump_json(indent=2))