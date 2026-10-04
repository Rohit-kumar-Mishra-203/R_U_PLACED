import argparse
import shutil
import sys
from pathlib import Path

from app.core.schema import ResumeFacts
from app.core.resume_parser import parse_resume
from app.core.latex_compiler import (
    TEMPLATE_DIR,
    TEMPLATE_NAME,
    render_latex,
    compile_to_pdf,
)

BASE_RESUME = Path("data/base_resume.tex")
FACTS_JSON = Path("data/resume_facts.json")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def preflight() -> None:
    """Fail early with a clear message if something is missing."""
    problems = []
    if not BASE_RESUME.exists():
        problems.append(f"Missing input resume: {BASE_RESUME}")
    if not (TEMPLATE_DIR / TEMPLATE_NAME).exists():
        problems.append(f"Missing template: {TEMPLATE_DIR / TEMPLATE_NAME}")
    if shutil.which("tectonic") is None:
        problems.append("`tectonic` not found on PATH (needed to compile the PDF)")
    if problems:
        print("Preflight failed:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)


def step_parse(reparse: bool) -> ResumeFacts:
    if FACTS_JSON.exists() and not reparse:
        print(f"[1/3] Loading cached facts from {FACTS_JSON} (use --reparse to redo)")
        return ResumeFacts.model_validate_json(FACTS_JSON.read_text(encoding="utf-8"))

    print("[1/3] Parsing resume with the LLM...")
    facts = parse_resume(read_text(BASE_RESUME))
    FACTS_JSON.write_text(facts.model_dump_json(indent=2), encoding="utf-8")
    print(f"      Saved {FACTS_JSON}")
    return facts


def print_summary(facts: ResumeFacts) -> None:
    summary_text = facts.summary.summary if facts.summary else ""
    n_exp_bullets = sum(len(e.bullets) for e in facts.experience)
    n_proj_bullets = sum(len(p.bullets) for p in facts.projects)

    print("\n--- Extraction check ---")
    print(f"Name            : {facts.personal_info.name}")
    print(f"Email           : {facts.personal_info.email}")
    print(f"Summary         : {'yes (' + str(len(summary_text)) + ' chars)' if summary_text else 'none'}")
    print(f"Experience      : {len(facts.experience)} entries, {n_exp_bullets} bullets")
    print(f"Projects        : {len(facts.projects)} entries, {n_proj_bullets} bullets")
    print(f"Skill categories: {len(facts.skills)} ({sum(len(s.items) for s in facts.skills)} items)")
    print(f"Education       : {len(facts.education)}")
    print(f"Certifications  : {len(facts.certifications)}")

    # Quick sanity warnings
    for e in facts.experience:
        if not e.bullets:
            print(f"  ! {e.id} ({e.company}) has no bullets")
        if e.end_date is None:
            print(f"  ! {e.id} ({e.company}) end_date is null (expected 'Present')")
    print("------------------------\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reparse", action="store_true", help="re-run LLM extraction")
    parser.add_argument("--name", default="resume", help="output file name (no extension)")
    args = parser.parse_args()

    preflight()

    facts = step_parse(args.reparse)
    print_summary(facts)

    print("[2/3] Rendering LaTeX from template...")
    tex_source = render_latex(facts)
    print(f"      Rendered {len(tex_source)} characters of LaTeX")

    print("[3/3] Compiling PDF with tectonic...")
    pdf_path = compile_to_pdf(tex_source, output_filename=args.name)
    print(f"\nDone. PDF written to: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()