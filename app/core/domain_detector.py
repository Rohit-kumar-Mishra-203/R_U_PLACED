from typing import List
from pydantic import BaseModel
from app.core.schema import ResumeFacts
from app.core.groq_client import invoke_structured

# Each keyword = one search request per career site, so keep this small.
MAX_KEYWORDS = 8


class SearchKeywords(BaseModel):
    keywords: List[str]


COMPACT_SCHEMA_EXAMPLE = '{"keywords": ["string", "string"]}'

PROMPT = """Based on this candidate's summary, roles, and skills, generate 50 to 80
job title search terms that would find genuinely relevant open roles for them.

Respond with ONLY a valid JSON object matching this exact shape, nothing else:

{schema}

Rules:
1. Use real, common job title phrasing (e.g. "Machine Learning Engineer").
2. Base this only on what's actually in their background below - don't guess
   at aspirational roles they have no evidence for.
3. Return at most 8 titles. No numbering, no markdown, no explanation.

Summary: {summary}

Most recent roles: {roles}

Top skills: {skills}
"""


def detect_search_keywords(facts: ResumeFacts) -> List[str]:
    roles = ", ".join(exp.role for exp in facts.experience[:3])
    skills = ", ".join(item for cat in facts.skills[:4] for item in cat.items[:5])
    prompt = PROMPT.format(
        schema=COMPACT_SCHEMA_EXAMPLE,
        summary=facts.summary or "",
        roles=roles,
        skills=skills,
    )
    result = invoke_structured(SearchKeywords, prompt)

    # Clean up regardless of what the model returned: dedupe, trim, hard cap
    seen, cleaned = set(), []
    for kw in result.keywords:
        kw = kw.strip()
        if kw and kw.lower() not in seen:
            seen.add(kw.lower())
            cleaned.append(kw)
    return cleaned[:MAX_KEYWORDS]


if __name__ == "__main__":
    with open("data/resume_facts.json", "r", encoding="utf-8") as f:
        facts = ResumeFacts.model_validate_json(f.read())
    print("Detected search keywords:", detect_search_keywords(facts))