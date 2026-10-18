import os
from dotenv import load_dotenv
from app.core.schema import ResumeFacts
from typing import cast
from langchain_groq import ChatGroq
from pydantic import SecretStr
import json

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=cast(SecretStr, os.getenv("GROQ_API_KEY")),
)

structured_llm = llm.with_structured_output(ResumeFacts, method="json_mode")


EXTRACTION_PROMPT = """You are extracting structured data from a resume.

Respond with ONLY a valid JSON object matching the required structure. Do not include any explanation, markdown formatting, or text outside the JSON.

Your output MUST match this exact JSON schema - use these EXACT field names,
nothing else (e.g. use "personal_info" not "profileSummary", "experience" not
"experiences", "skills" not "technicalSkills", keep "education" as a LIST even
if there's only one entry):

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