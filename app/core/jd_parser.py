import json

from app.core.schema import JDRequirements


EXTRACTION_PROMPT = """You are extracting structured requirements from a job description.

Respond with ONLY a valid JSON object matching the required structure. Do not
include any explanation, markdown formatting, or text outside the JSON.

Your output MUST match this exact JSON schema - use these EXACT field names,
nothing else:

<<SCHEMA>>

Rules:
1. Separate "must_have_skills" from "nice_to_have_skills" - do not mix these.
2. "tools_and_tech" should list only concrete named technologies.
3. "responsibilities" should be short phrases, not copied verbatim sentences.
4. "seniority_level" is an integer from 1 to 5:
   1 = intern/entry, 2 = junior (about 0-2 years), 3 = mid (about 3-5 years),
   4 = senior (about 5-8 years), 5 = staff/principal/lead.
   Infer it from stated years of experience, job title, and scope of
   responsibilities. Use null ONLY if there is no usable signal at all.
5. Do not add skills or requirements that aren't stated or clearly implied.
6. Never output null for string or list fields. Use "" for an unknown string
   (e.g. "company") and [] for an empty list.

Job description:
---
<<JD_TEXT>>
---
"""


def _build_schema_json() -> str:
    """JSON schema shown to the LLM, without fields we fill in ourselves."""
    schema = JDRequirements.model_json_schema()
    schema.get("properties", {}).pop("raw_text", None)
    if "required" in schema:
        schema["required"] = [f for f in schema["required"] if f != "raw_text"]
    return json.dumps(schema, indent=2)


def parse_jd(jd_text: str) -> JDRequirements:
    # .replace() instead of .format() so braces in the JD or schema can't break it
    prompt = EXTRACTION_PROMPT.replace("<<SCHEMA>>", _build_schema_json()).replace(
        "<<JD_TEXT>>", jd_text
    )
    return result


if __name__ == "__main__":
    sample_jd = """
    We are hiring a Machine Learning Engineer to join our NLP team.
    Requirements: 2+ years experience with Python and PyTorch, strong
    understanding of Transformer architectures, experience with LLM
    fine-tuning. Familiarity with LangChain/LangGraph and RAG systems
    is a plus. You will build and deploy production NLP pipelines,
    collaborate with the product team, and optimize model performance.
    """
    result = parse_jd(sample_jd)
    print(result.model_dump_json(indent=2))