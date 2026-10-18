import os
import json
from typing import cast
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr
from app.core.schema import ResumeFacts

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=cast(SecretStr, os.getenv("GROQ_API_KEY")),
    max_tokens=8192,
)

structured_llm = llm.with_structured_output(ResumeFacts, method="json_mode")

EXTRACTION_PROMPT = """You are extracting structured data from a resume.

Respond with ONLY a valid JSON object matching the required structure. Do not
include any explanation, markdown formatting, or text outside the JSON.

Your output MUST match this exact JSON schema - use these EXACT field names,
nothing else:

{schema}

Rules you must follow strictly:
1. Extract ONLY information explicitly present in the source text below.
   Do not infer, guess, or add anything that isn't there.
2. Preserve the original wording of bullet points exactly - do not paraphrase
   or improve them during extraction. This is raw extraction, not editing.
3. Split each bullet point into its own separate entry. Do not merge multiple
   bullets into one, and do not split one bullet into multiple entries.
4. Assign each experience entry an id like "exp1", "exp2" in order of appearance.
   Assign each project an id like "proj1", "proj2".
5. Assign each bullet an id combining its parent id and bullet number,
   e.g. "exp1_b1", "exp1_b2", "proj1_b1".
6. If a field genuinely isn't present (e.g. no GPA listed), leave it null
   rather than guessing.
7. For certifications, if an instructor name is mentioned (e.g. "Instructor:
   Krish Naik"), extract just the person's name into an "instructor" field -
   do not include the word "Instructor:" itself, that gets added back later.
8. For a current/ongoing role or bullet with no end date stated (e.g. "March
   2026 -- present"), set end_date to the literal string "Present" - never
   use null/None for end_date, even for ongoing positions.

Resume source:
---
{resume_text}
---
"""


def parse_resume(resume_text: str) -> ResumeFacts:
    schema_json = json.dumps(ResumeFacts.model_json_schema(), indent=2)
    prompt = EXTRACTION_PROMPT.format(schema=schema_json, resume_text=resume_text)
    result = cast(ResumeFacts, structured_llm.invoke(prompt))
    return result


if __name__ == "__main__":
    with open("data/base_resume.tex", "r", encoding="utf-8") as f:
        resume_text = f.read()

    facts = parse_resume(resume_text)

    with open("data/resume_facts.json", "w", encoding="utf-8") as f:
        f.write(facts.model_dump_json(indent=2))

    print("Extraction complete. Review data/resume_facts.json")
    print(f"Extracted {len(facts.experience)} experience entries, "
          f"{len(facts.projects)} projects, "
          f"{sum(len(s.items) for s in facts.skills)} skills.")