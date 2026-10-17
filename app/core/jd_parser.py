import os
from dotenv import load_dotenv
from app.core.schema import JDRequirements
from typing import cast
from langchain_groq import ChatGroq
from pydantic import SecretStr

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=cast(SecretStr, os.getenv("GROQ_API_KEY")),
)


structured_llm = llm.with_structured_output(JDRequirements, method="json_mode")

EXTRACTION_PROMPT = """You are extracting structured requirements from a job description.

Rules:
1. Separate "must_have_skills" (explicitly required/mandatory) from
   "nice_to_have_skills" (preferred, bonus, "a plus", optional) - do not
   mix these two lists.
2. "tools_and_tech" should list only concrete named technologies
   (e.g. "PyTorch", "Docker", "AWS") - not vague concepts.
3. "responsibilities" should be the actual day-to-day tasks/duties described,
   written as short phrases, not copied verbatim sentences.
4. Infer "seniority_level" only if there's a clear signal (years of experience
   required, title like "Senior", "Intern", etc). If unclear, leave it null.
5. Do not add skills or requirements that aren't stated or clearly implied
   in the text.

Job description:
---
{jd_text}
---
"""


def parse_jd(jd_text: str) -> JDRequirements:
    prompt = EXTRACTION_PROMPT.format(jd_text=jd_text)
    result = cast(JDRequirements, structured_llm.invoke(prompt))
    result.raw_text = jd_text  # ensure original text is preserved exactly
    return result


if __name__ == "__main__":
    # quick manual test - paste a real JD here to test against
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