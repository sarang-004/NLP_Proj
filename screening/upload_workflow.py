"""Request-scoped resume upload adapter for the existing screening scorer.

The adapter extracts text in memory, structures only evidence present in the
resume, and calls ``run_screening.score`` without persisting upload content.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from datetime import date
import json
import math
import re
from typing import Any, BinaryIO

import pandas as pd

from data_layer import COURSE_PATTERNS, NA, extract_degrees, structure_resume
from screening.decision_layer import evaluate
from screening.requirements import classify_jd_requirements, requirement_skill_names, skill_vocabulary


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_RESUME_BYTES = 12 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ResumeUploadError(ValueError):
    """A resume upload is empty, unsupported, or could not be extracted."""


def extract_uploaded_text(filename: str, stream: BinaryIO, max_bytes: int = MAX_RESUME_BYTES) -> str:
    """Extract text from a PDF, DOCX, or TXT stream without writing it to disk."""
    extension = Path(filename or "").suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ResumeUploadError("Unsupported file type. Upload PDF, DOCX, or TXT.")
    content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ResumeUploadError("File exceeds the 12 MB per-resume limit.")
    if not content:
        raise ResumeUploadError("The uploaded file is empty.")

    try:
        if extension == ".txt":
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = content.decode("cp1252")
        elif extension == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ResumeUploadError("PDF support is unavailable; install project requirements.") from exc
            text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
        else:
            try:
                from docx import Document
            except ImportError as exc:
                raise ResumeUploadError("DOCX support is unavailable; install project requirements.") from exc
            document = Document(BytesIO(content))
            paragraphs = [paragraph.text for paragraph in document.paragraphs]
            table_text = [cell.text for table in document.tables for row in table.rows for cell in row.cells]
            text = "\n".join(paragraphs + table_text)
    except ResumeUploadError:
        raise
    except Exception as exc:
        raise ResumeUploadError(f"Could not read {extension[1:].upper()} file: {exc}") from exc

    if not text.strip():
        raise ResumeUploadError("No readable text was found in the uploaded file.")
    return text


def _split_values(value: Any) -> list[str]:
    return [part.strip() for part in re.split(r"[|,;\n]+", str(value or "")) if part.strip()]


def _job_skill_vocabulary() -> list[str]:
    """Reuse the canonical project/common JD vocabulary for resume mentions."""
    return skill_vocabulary()


def extract_jd_skills(jd_text: str) -> list[str]:
    """Return criteria in qualification clauses, excluding responsibilities and mentions."""
    return requirement_skill_names({"job_description": jd_text})


def _extract_resume_skill_mentions(text: str) -> list[str]:
    matches = []
    for skill in _job_skill_vocabulary():
        match = re.search(r"(?<![\w])" + re.escape(skill) + r"(?![\w])", text, re.I)
        if match:
            matches.append((match.start(), skill))
    return list(dict.fromkeys(skill for _, skill in sorted(matches)))


def extract_preferred_qualification_evidence(text: str, requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find explicit resume evidence for preferred learning/experience requirements.

    Evidence is reported by type and never turned into a score here. In
    particular, a project or certificate is not treated as coursework or
    professional deployment experience.
    """
    records = [item for item in requirements if item.get("kind") == "preferred_qualification"]
    if not records:
        return []
    lines = str(text or "").splitlines()
    section = "resume"
    found = []
    for line in lines:
        content = line.strip()
        if not content:
            continue
        heading = re.sub(r"[^a-z]", "", content.casefold())
        if heading.startswith(("certification", "certificate")):
            section = "certifications"
        elif heading.startswith(("coursework", "relevantcourse", "academiccourse")):
            section = "coursework"
        elif heading.startswith(("project", "academicproject")):
            section = "projects"
        elif heading.startswith(("experience", "employment", "workexperience", "professionalexperience", "internship")):
            section = "professional experience"

        for requirement in records:
            topic = str(requirement.get("criterion", ""))
            if not topic:
                continue
            is_cloud = bool(re.search(r"\bcloud\b|\bAWS\b|\bAzure\b|\bGCP\b", topic, re.I))
            topic_pattern = r"\b(?:cloud|AWS|Azure|GCP)\b" if is_cloud else r"(?<!\w)" + re.escape(topic) + r"(?!\w)"
            if not re.search(topic_pattern, content, re.I):
                continue
            if section == "certifications" or re.search(r"\b(?:certified|certification|certificate)\b", content, re.I):
                evidence_type = "certification"
            elif section == "coursework" or re.search(r"\b(?:coursework|course|class)\b", content, re.I):
                evidence_type = "coursework"
            elif section == "projects" or re.search(r"\bproject\b", content, re.I):
                evidence_type = "project"
            elif section == "professional experience" and re.search(r"\b(?:deployed|deployment|production|professional|engineered|maintained)\b", content, re.I):
                evidence_type = "professional_experience"
            else:
                evidence_type = "mention_requires_review"
            item = {"requirement": topic, "evidence_type": evidence_type, "resume_section": section,
                    "resume_excerpt": content[:500], "scored": False}
            if item not in found:
                found.append(item)
    return found


def extract_jd_minimum_experience(jd_text: str) -> float | None:
    """Read explicit numeric years-of-experience requirements from JD text.

    For a stated range (for example, 2-4 years), use its lower bound. Wording
    without an explicit number is left unavailable for the scorer adapter.
    """
    text = re.sub(r"\s+", " ", str(jd_text or ""))
    unit = r"(?:years?|yrs?)"
    number = r"(\d+(?:\.\d+)?)"
    patterns = [
        # A range specifies its minimum at the lower bound.
        rf"\b{number}\s*(?:-|to)\s*\d+(?:\.\d+)?\s*{unit}\s*(?:of\s+)?(?:relevant\s+|professional\s+|work\s+)?experience\b",
        rf"\b(?:minimum|min\.?|minimum\s+of|at\s+least|a\s+minimum\s+of)\s*{number}\s*\+?\s*{unit}\s*(?:of\s+)?(?:relevant\s+|professional\s+|work\s+)?experience\b",
        rf"\b{number}\s*\+?\s*{unit}\s*['’]?\s*(?:of\s+)?(?:relevant\s+|professional\s+|work\s+)?experience\b",
        rf"\bexperience\s*(?:of\s*|:|-)\s*{number}\s*\+?\s*{unit}\b",
        rf"\b{number}\s*\+?\s*{unit}\s+(?:of\s+)?(?:professional\s+)?(?:experience\s+)?(?:developing|building|programming|working|engineering|administering)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1))
    return None


def make_upload_job(fields: dict[str, Any]) -> dict[str, Any]:
    """Build scorer inputs from explicit overrides or requirements in the JD."""
    jd_text = str(fields.get("job_description", "")).strip()
    degree_override = bool(str(fields.get("preferred_degree", "")).strip() or str(fields.get("accepted_degrees", "")).strip())
    requirement_context = classify_jd_requirements(jd_text)
    title = str(fields.get("job_title", "")).strip()
    skills_override = _split_values(fields.get("required_skills", ""))
    detected_skills = extract_jd_skills(jd_text)
    skills = skills_override or detected_skills
    skills_source = "user_override" if skills_override else ("job_description" if detected_skills else "not_detected")
    if not jd_text:
        raise ValueError("Enter the job description.")
    if not title:
        raise ValueError("Enter a job title or role label.")

    exp_raw = str(fields.get("minimum_experience", "")).strip()
    try:
        if exp_raw:
            minimum_experience = float(exp_raw)
            experience_source = "user_override"
        else:
            eligible_durations = [item.get("duration_years") for item in requirement_context["requirements"]
                                  if item.get("kind") == "experience_duration"
                                  and item.get("category") in {"explicit_minimum", "preferred"}
                                  and item.get("duration_years") is not None]
            detected_experience = min(eligible_durations) if eligible_durations else None
            minimum_experience = detected_experience if detected_experience is not None else 0.0
            experience_source = "job_description" if detected_experience is not None else "not_detected"
    except ValueError as exc:
        raise ValueError("Minimum experience must be a number of years.") from exc
    if not math.isfinite(minimum_experience):
        raise ValueError("Minimum experience must be a finite number of years.")
    if minimum_experience < 0:
        raise ValueError("Minimum experience cannot be negative.")

    degree_preferred = str(fields.get("preferred_degree", "")).strip()
    degree_accepted = _split_values(fields.get("accepted_degrees", ""))
    course_preferred = str(fields.get("preferred_course", "")).strip()
    course_accepted = _split_values(fields.get("accepted_courses", ""))
    if not degree_preferred and not degree_accepted:
        eligible_education = [item for item in requirement_context["requirements"]
                              if item["kind"] == "education" and item["category"] in {"explicit_minimum", "preferred"}]
        detected_degrees = list(dict.fromkeys(degree for item in eligible_education for degree in extract_degrees(item["source_text"])))
        bachelor_group = ["B.Tech", "B.E.", "BCA", "B.Com", "B.Sc", "B.A.", "BFA", "LLB", "B.A. LL.B.", "BBA LL.B."]
        master_group = ["M.Tech", "M.E.", "MCA", "MBA", "M.Com", "LLM", "M.Sc", "M.A.", "MMS"]
        for item in eligible_education:
            text = item["source_text"]
            if re.search(r"\bbachelor'?s?\s+(?:degree|qualification)\b", text, re.I):
                detected_degrees.extend(bachelor_group)
            if re.search(r"\bmaster'?s?\s+(?:degree|qualification)\b", text, re.I):
                detected_degrees.extend(master_group)
        detected_degrees = list(dict.fromkeys(detected_degrees))
        preferred_degrees = [degree for item in eligible_education if item["category"] == "preferred"
                             for degree in extract_degrees(item["source_text"])]
        for item in eligible_education:
            if item["category"] != "preferred":
                continue
            if re.search(r"\bbachelor'?s?\s+(?:degree|qualification)\b", item["source_text"], re.I):
                preferred_degrees.append("Bachelor's Degree")
            if re.search(r"\bmaster'?s?\s+(?:degree|qualification)\b", item["source_text"], re.I):
                preferred_degrees.append("Master's Degree")
        preferred_degrees = list(dict.fromkeys(preferred_degrees))
        if detected_degrees:
            if preferred_degrees:
                degree_preferred = preferred_degrees[0]
                degree_accepted = [degree for degree in detected_degrees if degree != degree_preferred]
            else:
                degree_accepted = detected_degrees
    if not course_preferred and not course_accepted:
        eligible_courses = [item for item in requirement_context["requirements"]
                            if item["kind"] == "course" and item["category"] in {"explicit_minimum", "preferred"}]
        found_courses = []
        for item in eligible_courses:
            for pattern, canonical in COURSE_PATTERNS:
                if re.search(pattern, item["source_text"], re.I):
                    found_courses.append((canonical, item["category"]))
                    break
        found_courses = list(dict.fromkeys(found_courses))
        preferred_courses = [course for course, category in found_courses if category == "preferred"]
        if preferred_courses:
            course_preferred = preferred_courses[0]
            course_accepted = [course for course, _ in found_courses if course != course_preferred]
        elif found_courses:
            course_accepted = [course for course, _ in found_courses]

    education_requirements = []
    for item in requirement_context["requirements"]:
        if item["kind"] != "education" or item["category"] not in {"explicit_minimum", "preferred"}:
            continue
        source = item["source_text"]
        level = "bachelor" if re.search(r"\bbachelor(?:'s)?\b", source, re.I) else "master" if re.search(r"\bmaster(?:'s)?\b", source, re.I) else None
        subjects = []
        for pattern, subject in ((r"computer science\s+and\s+engineering", "Computer Science and Engineering"),
                                 (r"computer science", "Computer Science"),
                                 (r"software engineering", "Software Engineering")):
            if re.search(pattern, source, re.I):
                subjects.append(subject)
        if level or subjects:
            education_requirements.append({"level": level, "subjects": subjects,
                                           "preference": "preferred" if item["category"] == "preferred" else "mandatory",
                                           "source_text": source})
    if degree_override:
        education_requirements = []

    # The raw JD drives semantic scoring. Skills and experience are parsed
    # deterministically from explicit JD mentions unless the user overrides.
    return {
        "job_id": "UPLOAD_SESSION",
        "job_title": title,
        "required_skills": "|".join(skills),
        "required_skills_list": skills,
        "required_skills_source": skills_source,
        "minimum_total_experience_years": minimum_experience,
        "experience_requirement_source": experience_source,
        "requirements": requirement_context,
        "job_description": jd_text,
        "preferred_degree": degree_preferred,
        "accepted_degrees": "|".join(degree_accepted),
        "preferred_course": course_preferred,
        "accepted_courses": "|".join(course_accepted),
        "education_requirements": education_requirements,
        "semantic_text_override": jd_text,
    }


def _explicit_skills_section(text: str) -> str:
    """Return a labeled skills section only, avoiding experience text bleed."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        heading = re.sub(r"[^a-z]", "", line.casefold())
        if heading in {"skills", "technicalskills", "keyskills"}:
            section = []
            for candidate in lines[index + 1:]:
                next_heading = re.sub(r"[^a-z]", "", candidate.casefold())
                if next_heading in {"tools", "technologies", "toolstechnologies", "workexperience", "experience", "education", "certifications"}:
                    break
                section.append(candidate)
            return "\n".join(section).strip()
    match = re.search(
        r"(?ims)^\s*(?:technical\s+|core\s+)?skills?\s*[:\-]\s*(.*?)(?=^\s*(?:(?:total\s+)?experience|employment|education|projects?|certifications?|summary|objective)\s*[:\-]|\Z)",
        text,
    )
    return match.group(1).strip() if match else ""


def _explicit_projects_section(text: str) -> str:
    """Extract text under a clearly labeled Projects section, bounded by the next section."""
    lines = str(text or "").splitlines()
    headings = {
        "projects", "project", "academicprojects", "personalprojects", "keyprojects",
        "experience", "workexperience", "professionalexperience", "employment",
        "education", "certifications", "certificates", "skills", "technicalskills",
        "summary", "objective", "achievements", "publications", "interests",
    }
    start = None
    for index, line in enumerate(lines):
        heading = re.sub(r"[^a-z]", "", line.casefold())
        if heading in {"projects", "project", "academicprojects", "personalprojects", "keyprojects"}:
            start = index + 1
            break
    if start is None:
        return ""
    content = []
    for line in lines[start:]:
        heading = re.sub(r"[^a-z]", "", line.casefold())
        if heading in headings:
            break
        if line.strip():
            content.append(re.sub(r"\s+", " ", line).strip())
    return "\n".join(content).strip()[:6000]


def _explicit_education_section(text: str) -> str:
    """Keep education credentials inside the resume's labeled education section."""
    lines = str(text or "").splitlines()
    headings = {
        "education", "educationalqualifications", "academicbackground", "academics",
        "projects", "project", "academicprojects", "personalprojects", "keyprojects",
        "experience", "workexperience", "professionalexperience", "employment", "internships",
        "skills", "technicalskills", "keyskills", "certifications", "certificates",
        "summary", "objective", "achievements", "publications", "interests",
    }
    for index, line in enumerate(lines):
        if re.sub(r"[^a-z]", "", line.casefold()) not in {"education", "educationalqualifications", "academicbackground", "academics"}:
            continue
        content = []
        for candidate in lines[index + 1:]:
            if re.sub(r"[^a-z]", "", candidate.casefold()) in headings:
                break
            if candidate.strip():
                content.append(re.sub(r"\s+", " ", candidate).strip())
        return "\n".join(content).strip()[:6000]
    return ""


def extract_resume_name(text: str) -> str | None:
    """Read a clearly labeled or isolated name near the resume header only."""
    lines = [re.sub(r"\s+", " ", line).strip(" \t|•") for line in str(text or "").splitlines()]
    lines = [line for line in lines[:12] if line]
    excluded = {"resume", "curriculum", "vitae", "candidate", "profile", "contact", "summary", "professional", "experience", "education", "skills", "technology", "software", "application", "mobile", "quality", "assurance", "testing", "engineer", "developer", "analyst", "manager", "tester", "consultant", "administrator"}
    for line in lines:
        match = re.match(r"^(?:name)\s*[:\-]\s*(.+)$", line, re.I)
        if match:
            line = match.group(1).strip()
            break
    else:
        line = next((item for item in lines
                     if not re.search(r"@|https?://|\+?\d[\d\s().-]{7,}", item)
                     and 2 <= len(item.split()) <= 5
                     and not any(token.casefold().strip(".,:") in excluded for token in item.split())), "")

    tokens = line.split()
    if not 2 <= len(tokens) <= 5 or any(token.casefold().strip(".,:") in excluded for token in tokens):
        return None
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", token) for token in tokens):
        return None
    return line


def structure_uploaded_resume(text: str, resume_id: str, as_of: date | None = None) -> dict[str, Any]:
    """Use the existing conservative data-layer extractors on resume text."""
    section_skills = _explicit_skills_section(text)
    detected_skill_mentions = _extract_resume_skill_mentions(text)
    skills_raw = "\n".join(part for part in (section_skills, "|".join(detected_skill_mentions)) if part)
    education_section = _explicit_education_section(text)
    evaluation_date = as_of or date.today()
    row = structure_resume({
        "resume_id": resume_id,
        "name": NA,
        "college": NA,
        "city": NA,
        "resume_text": text,
        "education_raw": education_section or text,
        "skills_raw": skills_raw,
        "experience_raw": text,
    }, as_of=evaluation_date)
    row["resume_id"] = resume_id
    row["job_id"] = "UPLOAD_SESSION"
    project_section = _explicit_projects_section(text)
    if project_section:
        row["project_evidence_text"] = project_section
        row["projects_extracted"] = project_section
        row["projects"] = project_section
    parsed_experience = row.get("total_experience_years", NA)
    summary = re.search(
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience\b",
        text,
        re.I,
    )
    if summary:
        row["reported_total_experience_years"] = float(summary.group(1))
    else:
        row["reported_total_experience_years"] = None
    if parsed_experience in ("", NA):
        # Accept an explicitly stated profile summary such as "7+ years of
        # experience" when there are no parseable employment date ranges.
        if summary:
            parsed_experience = str(float(summary.group(1)))
            row["experience_source"] = "explicit_profile_summary"
    if parsed_experience not in ("", NA):
        row["total_experience_years"] = parsed_experience
        row["total_experience_years_final"] = parsed_experience
    return row


def _evidence_completeness(result: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Summarize evidence coverage without equating silence with a qualification gap."""
    criteria = []
    skill_items = result.get("skill_evidence", [])
    for item in skill_items:
        criteria.append({"factor": "skills", "criterion": item.get("requirement", ""),
                         "status": item.get("evidence_status", "not_demonstrated")})

    degree = result.get("degree_course_evidence", {})
    profile = degree.get("profile", {})
    education_required = bool(job.get("education_requirements") or job.get("preferred_degree")
                              or job.get("accepted_degrees") or job.get("preferred_course")
                              or job.get("accepted_courses"))
    education_status = "not_applicable"
    if education_required:
        raw_status = str(profile.get("status", degree.get("degree_status", ""))).casefold()
        if degree.get("degree_score") is None and degree.get("course_score") is None:
            education_status = "unavailable"
        elif "mismatch" in raw_status:
            education_status = "explicitly_mismatched"
        elif degree.get("degree_score") == 0 and degree.get("course_score") is None:
            education_status = "not_demonstrated"
        elif any(value is not None and 0 < value < 1 for value in
                 (degree.get("degree_score"), degree.get("course_score"))):
            education_status = "partially_demonstrated"
        else:
            education_status = "demonstrated"
        criteria.append({"factor": "education_course", "criterion": "JD education/course criterion",
                         "status": education_status})

    experience_required = bool(job.get("minimum_total_experience_years", 0) > 0 or any(
        item.get("kind") == "experience_duration" and item.get("category") in {"explicit_minimum", "preferred"}
        for item in job.get("requirements", {}).get("requirements", [])
    ))
    experience_score = result.get("component_scores", {}).get("experience", result.get("experience_score"))
    if experience_required:
        if experience_score is None:
            experience_status = "unavailable"
        elif experience_score >= 1:
            experience_status = "demonstrated"
        elif experience_score > 0:
            experience_status = "partially_demonstrated"
        else:
            experience_status = "measured_below_requirement"
        criteria.append({"factor": "experience", "criterion": "JD experience criterion",
                         "status": experience_status})
    else:
        experience_status = "not_applicable"

    projects_status = "demonstrated_evidence_present" if result.get("projects_available") else "unavailable"
    semantic_status = "aggregate_signal_available" if result.get("semantic_score") is not None else "unavailable"
    evidence_found = sum(item["status"] not in {"not_demonstrated", "unavailable"} for item in criteria)
    criterion_count = len(criteria)
    unavailable_factors = list(result.get("excluded_factors", []))
    if not unavailable_factors:
        unavailable_factors = [factor for factor, score in result.get("component_scores", {}).items() if score is None]
    return {
        "scored_criterion_count": criterion_count,
        "criteria_with_resume_evidence": evidence_found,
        "criteria_not_demonstrated": sum(item["status"] == "not_demonstrated" for item in criteria),
        "criteria_unavailable": sum(item["status"] == "unavailable" for item in criteria),
        "criterion_evidence_coverage": round(evidence_found / criterion_count, 4) if criterion_count else None,
        "criteria": criteria,
        "factor_status": {
            "skills": ("unavailable" if not skill_items else
                       "not_demonstrated" if all(item.get("evidence_status") == "not_demonstrated" for item in skill_items) else
                       "partially_demonstrated" if any(item.get("evidence_status") != "demonstrated" for item in skill_items) else "demonstrated"),
            "education_course": education_status,
            "experience": experience_status,
            "projects": projects_status,
            "semantic": semantic_status,
        },
        "unavailable_factors": unavailable_factors,
        "coverage_note": "Coverage counts whether resume evidence was extracted for scored criteria; it is not a fit score. Not demonstrated means no supporting mention was found, not proof the candidate lacks the qualification. Semantic similarity is an aggregate signal, not qualification evidence.",
    }


def evaluate_uploads(uploads: list[tuple[str, str]], job: dict[str, Any], model: Any) -> list[dict[str, Any]]:
    """Score uploaded resumes and rank them; optional third tuple item is the filename label."""
    evaluated = []
    for upload in uploads:
        resume_id, text = upload[:2]
        filename_label = str(upload[2]).strip() if len(upload) > 2 else ""
        extracted_name = extract_resume_name(text)
        filename_name = re.sub(r"(?i)^(?:resume|cv|curriculum vitae)[\s_-]*", "", filename_label).strip(" _-")
        compact_name = lambda value: re.sub(r"[^a-z]", "", value.casefold())
        filename_corrects_spacing = bool(
            extracted_name and filename_name
            and compact_name(extracted_name) == compact_name(filename_name)
        )
        if filename_corrects_spacing:
            candidate_name = filename_name
            name_source = "filename_confirmed"
        else:
            candidate_name = extracted_name or filename_label or resume_id
            name_source = "resume" if extracted_name else "filename" if filename_label else "fallback_id"
        row = pd.Series(structure_uploaded_resume(text, resume_id))
        result = evaluate(row, pd.Series(job), model, upload_screening=True)
        missing_context = []
        if row.get("total_experience_years", NA) in ("", NA):
            missing_context.append("experience")
        if row.get("degree", NA) in ("", NA):
            missing_context.append("degree")
        if row.get("course", NA) in ("", NA):
            missing_context.append("course")
        if result["skill_score"] is None:
            missing_context.append("skill requirements")
        if result["experience_score"] is None and job.get("minimum_total_experience_years", 0) > 0:
            requirements = job.get("requirements", {}).get("requirements", [])
            technology_specific = any(
                item.get("kind") == "experience_duration"
                and item.get("criterion") != "total professional experience"
                and item.get("category") in {"explicit_minimum", "preferred"}
                for item in requirements
            )
            missing_context.append("technology-specific experience duration" if technology_specific else "total experience evidence")

        effective_weights = {
            "skills": result["effective_skill_weight"],
            "experience": result["effective_experience_weight"],
            "education_course": result["effective_education_course_weight"],
            "projects": result["effective_project_weight"],
            "semantic": result["effective_semantic_weight"],
        }
        component_scores = {
            "skills": result["skill_score"],
            "experience": result["experience_score"],
            "education_course": result["education_course_score"],
            "projects": result["project_score"],
            "semantic": result["semantic_score"],
        }
        contributions = {
            "skills": result["skill_contribution"],
            "experience": result["experience_contribution"],
            "education_course": result["education_course_contribution"],
            "projects": result["project_contribution"],
            "semantic": result["semantic_contribution"],
        }
        degree_course_context = {
            "degree": result["degree"],
            "degree_score": result["degree_score"],
            "degree_status": result["degree_status"],
            "course": result["course"],
            "course_score": result["course_score"],
            "course_status": result["course_status"],
            "profile": row.get("_education_evidence", {}),
        }
        experience_detail = result.get("experience_evidence", {})
        if isinstance(experience_detail, str):
            try:
                experience_detail = json.loads(experience_detail)
            except (ValueError, TypeError):
                experience_detail = {}
        if experience_detail.get("experience_requirement_type") == "relevant_role":
            relevant_months = experience_detail.get("relevant_role_months")
            total_months = experience_detail.get("total_professional_months")
            internship_months = experience_detail.get("internship_months")
            required_months = experience_detail.get("required_months")
            experience_explanation = (
                f"Total dated professional experience was {total_months if total_months is not None else 'unavailable'} months "
                f"(including {internship_months if internship_months is not None else 0} internship months). "
                f"JD-relevant experience was {relevant_months} months / {required_months:g} required months "
                f"= {result['experience_score']:.4f} (capped at 1.0); "
                "Only professional or internship roles with matching JD activity evidence count; "
                "student leadership, projects, and unrelated or unclassified roles remain separate."
                if relevant_months is not None else
                f"Total dated professional experience was {total_months if total_months is not None else 'unavailable'} months "
                f"(including {internship_months if internship_months is not None else 0} internship months), but JD-relevant duration is unavailable. "
                "Experience score is unavailable because no dated professional or internship role "
                "matched the JD activity. Extracted role dates and categories are retained for review."
            )
        elif experience_detail.get("experience_requirement_type") == "evidence_only_no_jd_duration":
            experience_explanation = (
                f"Resume experience evidence is present ({experience_detail.get('total_professional_months') or 0} dated professional months, "
                f"including {experience_detail.get('internship_months') or 0} internship months), but the JD specifies no duration to score against. "
                "The evidence is displayed without assigning an experience score."
            )
        elif experience_detail.get("experience_requirement_type") == "technology_specific":
            experience_explanation = (
                "Technology-specific experience requires explicit resume text tying duration to the named "
                "technology and activity; role dates and skills-list mentions alone do not establish it."
            )
        elif result["experience_score"] is None:
            experience_explanation = "Experience score is unavailable because no reliable total professional duration was extracted."
        else:
            experience_explanation = (
                f"Total professional experience uses {experience_detail.get('total_professional_months')} dated months "
                "from full-time, part-time, contract/freelance, and internship roles; leadership, projects, and "
                "unclassified roles are excluded."
            )
        formula_terms = [
            f"{name} contribution {contributions[name]:.6f}"
            for name in component_scores if component_scores[name] is not None and effective_weights[name] > 0
        ]
        evaluated.append({
            "resume_id": resume_id,
            "candidate_name": candidate_name,
            "candidate_name_source": name_source,
            "final_score": result["final_score"],
            "score_methodology_version": result["score_methodology_version"],
            "component_scores": component_scores,
            "configured_weights": result["configured_weights"],
            "skill_score_calculation": result.get("skill_score_calculation", {}),
            "effective_weights": effective_weights,
            "weighted_contributions": contributions,
            "contribution_sum": result["contribution_sum"],
            "score_denominator_weight": result["score_denominator_weight"],
            "excluded_factors": [name for name, value in component_scores.items() if value is None],
            "matched_skills": result["matched_skills"],
            "missing_skills": result["missing_skills"],
            "experience": {
                "years": row.get("total_experience_years", NA),
                "professional_months": row.get("total_professional_experience_months"),
                "months_by_category": row.get("experience_months_by_category", {}),
                "records": row.get("experience_records", []),
                "duration_convention": row.get("experience_duration_convention", "inclusive_calendar_months"),
                "reported_years": row.get("reported_total_experience_years"),
                "source": row.get("experience_source"),
                "as_of": row.get("experience_as_of"),
                "required_years": job["minimum_total_experience_years"],
                "status": result["experience_status"],
                "evidence": result["experience_evidence"],
                "available": result["experience_score"] is not None,
                "technology_specific": result.get("experience_evidence"),
            },
            "degree_course_evidence": degree_course_context,
            "skill_evidence": result.get("skill_evidence", []),
            "requirement_context": result.get("jd_requirement_context", {}),
            "preferred_qualification_evidence": extract_preferred_qualification_evidence(
                str(row.get("resume_text", "")), job.get("requirements", {}).get("requirements", [])
            ),
            "project_evidence": result["project_evidence"],
            "project_similarity_evidence": result["project_similarity_evidence"],
            "projects_available": result["projects_available"],
            "semantic_evidence": result["semantic_evidence"],
            "unavailable_evidence": missing_context,
            "evidence_completeness": _evidence_completeness({
                "skill_evidence": result.get("skill_evidence", []),
                "skill_score_calculation": result.get("skill_score_calculation", {}),
                "degree_course_evidence": degree_course_context,
                "component_scores": component_scores,
                "excluded_factors": [name for name, value in component_scores.items() if value is None],
                "projects_available": result.get("projects_available"),
                "semantic_score": result.get("semantic_score"),
            }, job),
            "formula": " + ".join(formula_terms) + f" = {result['final_score']:.6f}; factors without a scoreable JD criterion or sufficient evidence add no numeric contribution",
            "explanation": (
                f"Score {result['final_score']:.4f} uses methodology {result['score_methodology_version']}. "
                f"Skills evidence was found for {result['matched_skill_count']} of {result['required_skill_count']} scored JD criteria; an absent mention means not demonstrated in the supplied resume. "
                "Education and course share one weighted component; only supplied JD criteria are included."
                f" {experience_explanation}"
            ),
        })
    evaluated.sort(key=lambda item: (-item["final_score"], item["resume_id"]))
    for rank, item in enumerate(evaluated, start=1):
        item["rank"] = rank
    return evaluated
