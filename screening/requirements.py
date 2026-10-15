"""Deterministic, evidence-preserving classification of JD statements."""
from __future__ import annotations

import re
import csv
from typing import Any
from pathlib import Path

METHOD_VERSION = "evidence-v3"
# Upload screening uses differentiated required/preferred skill weights. The
# audit and legacy batch path retain evidence-v3 unless explicitly migrated.
UPLOAD_METHOD_VERSION = "evidence-v4"
SKILL_TERMS = (
    "AWS", "Ruby", "Java", "Python", "SQL", "NoSQL", "Databases", "Git", "Jenkins",
    "SVN", "Azure", "GCP", "DevOps", "Cloud", "Docker", "Kubernetes", "Terraform",
    "Automation", "Selenium", "Appium", "RestAssured", "Postman", "GraphQL", "REST API",
    "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust", "MongoDB", "PostgreSQL",
)
SKILL_ALIASES = {"Amazon Web Services": "AWS"}
ADDITIONAL_SKILL_TERMS = {
    "Azure", "GCP", "Ruby", "Go", "Rust", "Kotlin", "Swift", "TypeScript", "PHP", "Perl", "Scala",
    "C++", "Bash", "PowerShell", "Docker", "Kubernetes", "Terraform", "Ansible", "CI/CD", "DynamoDB",
    "Redis", "Kafka", "RestAssured", "JUnit", "SVN",
}


def skill_vocabulary() -> list[str]:
    terms = set(SKILL_TERMS) | ADDITIONAL_SKILL_TERMS | set(SKILL_ALIASES)
    source = Path(__file__).resolve().parents[1] / "data" / "structured_job_descriptions.csv"
    try:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for column in ("required_skills", "preferred_skills"):
                    terms.update(value.strip() for value in str(row.get(column, "")).split("|") if value.strip() and value.strip().upper() != "NA")
    except (OSError, UnicodeError, csv.Error):
        pass
    return sorted(terms, key=lambda value: (-len(value), value.casefold()))
FLEXIBLE_RE = re.compile(r"\b(?:ideal(?:ly)?|preferred|bonus|desirable|advantage|a plus|is a plus|exposure to|familiarity with|coursework in|certification in|not (?:need|required to)|need not|not expected to|required to meet every|without meeting every|not all)\b", re.I)
GLOBAL_FLEXIBLE_RE = re.compile(r"\b(?:may not meet (?:all|every)|may not all be met|do not necessarily have all|not (?:need|required to|expected to)\b|need not\b|without meeting every|required to meet every|not all\b)", re.I)
RESPONSIBILITY_RE = re.compile(r"\b(?:responsibilit(?:y|ies)|you will|duties include|responsible for|develop and deploy|design and implement|build and maintain|monitor(?:ing)?|deployment tasks?)\b", re.I)
ACTIVITY_RE = re.compile(r"\b(develop(?:ing|ment)?|program(?:ming|mer)?|deploy(?:ing|ment)?|administer(?:ing)?|design(?:ing)?|build(?:ing)?|work(?:ing)?|engineer(?:ing)?)\b", re.I)


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _terms(text: str) -> list[tuple[int, str]]:
    found = []
    for term in skill_vocabulary():
        plural = r"s?" if term.casefold().endswith("api") else ""
        m = re.search(r"(?<![\w])" + re.escape(term) + plural + r"(?![\w])", text, re.I)
        if m:
            found.append((m.start(), SKILL_ALIASES.get(term, term)))
    # Common JD phrasing for test automation should map to the existing
    # canonical Automation skill instead of silently dropping the criterion.
    for pattern in (r"\bautomated\s+testing\b", r"\btesting\s+automation\b"):
        match = re.search(pattern, text, re.I)
        if match:
            found.append((match.start(), "Automation"))
    return sorted(found)


def classify_jd_requirements(text: str) -> dict[str, Any]:
    """Return statements with source text and conservative requirement labels.

    Responsibilities and uncontextualized mentions are retained for context but
    excluded from the score. A global "ideal / need not meet all" qualifier
    downgrades qualification bullets to preferred/flexible.
    """
    source = re.sub(r"\r", "", _clean(text))
    # A single preferred item must not downgrade unrelated explicit minimums.
    # Only broad wording about not meeting every qualification applies globally.
    flexible = bool(GLOBAL_FLEXIBLE_RE.search(source))
    records: list[dict[str, Any]] = []
    # Preserve section headings so qualification lines inherit their context;
    # PDF extraction often wraps one bullet across multiple physical lines.
    section = "general"
    statements = []
    for raw in source.splitlines():
        heading = re.sub(r"[^a-z]", "", raw.casefold())
        if heading.startswith("responsibilities") or heading in {"responsibility", "duties"}:
            section = "responsibility"
            if re.fullmatch(r"\s*(?:responsibilities|responsibility|duties)(?:\s+for\s+[^:]*)?\s*:?[\s]*", raw, re.I):
                continue
            raw = re.sub(r"^\s*(?:responsibilities|responsibility|duties)(?:\s+for\s+[^:]*)?\s*:\s*", "", raw, flags=re.I)
        if heading.startswith("qualifications") or heading.startswith("requirements"):
            section = "qualification"
            if re.fullmatch(r"\s*(?:qualifications|requirements)(?:\s+for\s+[^:]*)?\s*:?[\s]*", raw, re.I):
                continue
            raw = re.sub(r"^\s*(?:qualifications|requirements)(?:\s+for\s+[^:]*)?\s*:\s*", "", raw, flags=re.I)
        for part in re.split(r"(?<=[.!?])\s+", raw):
            if part.strip(): statements.append((part, section))
    for raw, section in statements:
        statement = re.sub(r"\s+", " ", raw).strip(" \t-*•")
        if not statement:
            continue
        responsibility = section == "responsibility" or bool(RESPONSIBILITY_RE.search(statement))
        is_qualification = section == "qualification" or bool(re.search(r"\b(?:qualif|require|experience|proficien|skill|knowledge|must|minimum|at least|degree|bachelor|master|course|certification|certificate|exposure|familiarity|advantage|desirable|preferred|ideal(?:ly)?)\w*\b", statement, re.I))
        duration_match = re.search(r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", statement, re.I)
        # Coursework/certification statements describe learning evidence, not
        # standalone technical skill requirements. Keep them out of the skills
        # factor; course extraction below can still retain an explicit course.
        learning_context = bool(re.search(r"\b(?:coursework|certification|certificate|training)\b", statement, re.I))
        technologies = [] if learning_context else [term for _, term in _terms(statement)]
        is_explicit = bool(re.search(r"\b(?:must|required|minimum|at least|requirement)\b", statement, re.I))
        if responsibility:
            category = "responsibility"
        elif is_qualification:
            category = "preferred" if flexible or re.search(r"\b(?:preferred|ideal(?:ly)?|bonus|desirable|advantage|a plus|is a plus|exposure to|familiarity with|coursework in|certification in)\b", statement, re.I) else ("explicit_minimum" if is_explicit else "uncertain")
        else:
            category = "general_mention"
        if duration_match and not responsibility:
            category = "preferred" if flexible or re.search(r"\b(?:preferred|ideal(?:ly)?|bonus|desirable|a plus|is a plus)\b", statement, re.I) else ("explicit_minimum" if is_explicit or "+" in duration_match.group(0) else "uncertain")
        activity = ACTIVITY_RE.search(statement)
        learning_match = re.search(
            r"\b(?P<types>coursework|courses?|certifications?|certificates?|training)(?:\s+or\s+(?P<other>coursework|courses?|certifications?|certificates?|training))*\s+(?:in|on|with)\s+(?P<topic>[^,.!?]+)",
            statement, re.I,
        )
        if not learning_match:
            learning_match = re.search(
                r"\b(?P<topic>(?:cloud(?:\s+[\w-]+){0,3}|AWS|Azure|GCP|[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2}))\s+(?P<types>coursework|courses?|certifications?|certificates?|training)\b",
                statement, re.I,
            )
        if learning_match:
            types = re.findall(r"coursework|courses?|certifications?|certificates?|training", learning_match.group(0), re.I)
            topic = re.sub(r"\s+(?:is|would be|are)\s+(?:a\s+)?(?:plus|advantage|preferred|desirable).*$", "", learning_match.group("topic"), flags=re.I).strip()
            if topic:
                records.append({
                    "kind": "preferred_qualification",
                    "criterion": topic,
                    "category": "preferred" if category == "preferred" else category,
                    "preference_status": "preferred" if category == "preferred" else "unspecified",
                    "evidence_types": list(dict.fromkeys(t.casefold().rstrip("s") for t in types)),
                    "evidence_type": "learning_or_certification",
                    "confidence": "medium",
                    "duration_years": None,
                    "activity": "",
                    "source_text": statement,
                })
        # Individual technology requirements share the original sentence. The
        # score can then ask for evidence per technology without assigning years.
        if technologies:
            for term in technologies:
                records.append({
                    "kind": "experience_duration" if duration_match else "skill",
                    "criterion": term,
                    "category": category,
                    "duration_years": float(duration_match.group(1)) if duration_match else None,
                    "activity": activity.group(1).lower() if activity else "",
                    "source_text": statement,
                })
                if duration_match:
                    records.append({"kind": "skill", "criterion": term, "category": category,
                                    "duration_years": None, "activity": "", "source_text": statement})
        elif duration_match:
            records.append({
                "kind": "experience_duration", "criterion": "total professional experience",
                "category": category, "duration_years": float(duration_match.group(1)),
                "activity": activity.group(1).lower() if activity else "", "source_text": statement,
            })
        if re.search(r"\b(?:degree|bachelor|master|ph\.d|b\.tech|b\.e\.)\b", statement, re.I):
            records.append({"kind": "education", "criterion": "degree", "category": category, "duration_years": None, "activity": "", "source_text": statement})
        # A degree subject ("bachelor's degree in Computer Science") is not a
        # separate course requirement. Only explicit course/major/specialization
        # wording creates a course factor.
        if not learning_match and re.search(r"\b(?:course(?:work)?|speciali[sz]ation|major in)\b", statement, re.I):
            records.append({"kind": "course", "criterion": "course/specialization", "category": category, "duration_years": None, "activity": "", "source_text": statement})
        if not technologies and not duration_match and not re.search(r"\b(?:degree|bachelor|master|ph\.d|course|speciali[sz]ation)\b", statement, re.I):
            records.append({"kind": "statement", "criterion": "", "category": category, "duration_years": None, "activity": "", "source_text": statement})
    return {"jd_flexible_qualifications": flexible, "requirements": records}


def requirement_skill_names(job: Any) -> list[str]:
    """Scoring-eligible skills from explicit structured fields or JD clauses."""
    manual = _clean(job.get("required_skills", ""))
    if manual and manual.casefold() not in {"na", "n/a", "none", "nan"}:
        values = re.split(r"[|,;\n]+", manual)
        return list(dict.fromkeys(v.strip() for v in values if v.strip()))
    parsed = job.get("requirements")
    if isinstance(parsed, str):
        import json
        try:
            parsed = json.loads(parsed)
        except (ValueError, TypeError):
            parsed = None
    if isinstance(parsed, dict):
        records = parsed.get("requirements", [])
    elif isinstance(parsed, list):
        records = parsed
    else:
        description = _clean(job.get("job_description", "")) or _clean(job.get("semantic_text_override", ""))
        records = classify_jd_requirements(description)["requirements"]
    return list(dict.fromkeys(
        str(item["criterion"]) for item in records
        if item.get("kind") == "skill" and item.get("category") in {"explicit_minimum", "preferred"} and item.get("criterion")
    ))

