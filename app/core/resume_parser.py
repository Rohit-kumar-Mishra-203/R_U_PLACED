import os
from typing import cast
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from app.core.schema import ResumeFacts, ResumeFactsPart1, ResumeFactsPart2

load_dotenv()

llm=ChatGroq(model="openai/gpt-oss-120b") #type: ignore


part1_llm = llm.with_structured_output(ResumeFactsPart1)
part2_llm = llm.with_structured_output(ResumeFactsPart2)

PART1_PROMPT = """You are extracting structured data from a resume - specifically
the personal info, skills, education, and certifications ONLY.
Do NOT extract experience or projects - those are handled separately.

Rules:
1. Extract ONLY information explicitly present in the source text below.
   Do not infer, guess, or add anything that isn't there.
2. If a field genuinely isn't present, leave it null rather than guessing.
3. For certifications, if an instructor name is mentioned, extract just the
   person's name into an "instructor" field.

Resume source:
---
{resume_text}
---
"""



PART2_PROMPT = """You are extracting structured data from a resume - specifically
the summary, work experience and projects ONLY, including every bullet point.
Do NOT extract personal info, skills, education, or certifications - those
are handled separately.

Rules:
1. Extract ONLY information explicitly present in the source text below.
   Do not infer, guess, or add anything that isn't there.
2. Preserve the original wording of the summary and of every bullet point
   exactly - do not paraphrase, shorten, or rewrite.
3. Assign each experience entry an id like "exp1", "exp2" in order of
   appearance. Assign each project an id like "proj1", "proj2". Assign each
   bullet an id combining its parent id and bullet number, e.g. "exp1_b1".
4. Split each bullet point into its own separate entry - do not merge
   multiple bullets into one, and do not split one bullet into multiple.
5. For a current/ongoing role with no end date stated, set end_date to the
   literal string "Present" - never null.
6. The summary is the profile/objective/about-me paragraph near the top of
   the resume (it may be headed "Summary", "Profile", "Objective", or
   "About Me"). Extract it as a single string, keeping it as one block of
   text - do not split it into multiple entries or merge it with other
   sections.
7. If the resume has no summary section, set summary to an empty string ""
   - do not write one yourself from the experience or other sections.

Resume source:
---
{resume_text}
---
"""



def parse_resume(resume_text: str) -> ResumeFacts:
    part1_prompt = PART1_PROMPT.format(resume_text=resume_text)
    part1 = cast(ResumeFactsPart1, part1_llm.invoke(part1_prompt))

    part2_prompt = PART2_PROMPT.format(resume_text=resume_text)
    part2 = cast(ResumeFactsPart2, part2_llm.invoke(part2_prompt))

    return ResumeFacts(
        personal_info=part1.personal_info,
        skills=part1.skills,
        education=part1.education,
        certifications=part1.certifications,
        summary=part2.summary,
        experience=part2.experience,
        projects=part2.projects,
    )


if __name__ == "__main__":
    try:
        with open("data/base_resume.tex", "r", encoding="utf-8") as f:
            resume_text = f.read()
    except UnicodeDecodeError:
        with open("data/base_resume.tex", "r", encoding="latin-1") as f:
            resume_text = f.read()

    facts = parse_resume(resume_text)

    with open("data/resume_facts.json", "w", encoding="utf-8") as f:
        f.write(facts.model_dump_json(indent=2))

    print("Extraction complete. Review data/resume_facts.json")
    print(f"Extracted {len(facts.experience)} experience entries, "
          f"{len(facts.projects)} projects, "
          f"{sum(len(s.items) for s in facts.skills)} skills.")