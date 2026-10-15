import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.schema import ResumeFacts
from typing import cast


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

structured_llm = llm.with_structured_output(ResumeFacts)

EXTRACTION_PROMPT = """You are extracting structured data from a resume.

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
    prompt = EXTRACTION_PROMPT.format(resume_text=resume_text)
    result = structured_llm.invoke(prompt)
    return cast(ResumeFacts, result)


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