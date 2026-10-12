import os
from typing import Type, TypeVar, cast
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr, BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)

_raw_keys = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
]
GROQ_KEYS = [k for k in _raw_keys if k]





def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "rate_limit" in msg or "429" in msg or "quota" in msg



    if last_error is not None:
        raise last_error
    raise RuntimeError("All Groq API keys exhausted, but no error was captured.")