from pydantic import BaseModel, Field
from typing import List, Optional


class PersonalInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None


class SkillCategory(BaseModel):
    category: str          # e.g. "Languages", "Frameworks", "Tools", "Concepts"
    items: List[str]       # e.g. ["Python", "SQL"]


class ExperienceBullet(BaseModel):
    id: str                   # unique id, e.g. "exp1_b1" - used for citation/traceability
    text: str                 # the raw bullet text, unedited, from your original resume
    keywords: List[str] = []  # optional: pre-extracted keywords for faster matching


class Experience(BaseModel):
    id: str                   # e.g. "exp1"
    company: str
    role: str
    start_date: str
    end_date: str              # "Present" if current
    location: Optional[str] = None
    bullets: List[ExperienceBullet]


class Project(BaseModel):
    id: str                    # e.g. "proj1"
    name: str
    tech_stack: List[str]
    bullets: List[ExperienceBullet]
    link: Optional[str] = None


class Education(BaseModel):
    degree: str
    institution: str
    start_date: str
    end_date: str
    coursework: List[str] = []
    gpa: Optional[str] = None


class Certification(BaseModel):
    name: str
    issuer: str
    date: Optional[str] = None


class ResumeFactsPart1(BaseModel):
    """Smaller portion of resume data - used to split extraction into two
    calls so each stays well within Groq's token limits."""
    personal_info: PersonalInfo
    summary: Optional[str] = None
    skills: List[SkillCategory]
    education: List[Education]
    certifications: List[Certification] = []


class ResumeFactsPart2(BaseModel):
    """Larger portion of resume data (bullets live here) - the other half
    of the split extraction."""
    experience: List[Experience]
    projects: List[Project]


class ResumeFacts(BaseModel):
    """
    The single source of truth for everything downstream.
    ...
    """
    personal_info: PersonalInfo
    summary: Optional[str] = None
    skills: List[SkillCategory]
    experience: List[Experience]
    projects: List[Project]
    education: List[Education]
    certifications: List[Certification] = []