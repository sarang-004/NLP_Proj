"""Conservative resume/JD structuring and counterfactual utilities.

Run from any directory with ``python data_layer.py``. Source datasets are read
only; generated artifacts are written under data/ and validation/.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
NA = "NA"
DEGREES = [
    (r"\bB\s*\.\s*A\s*\.\s*LL\s*\.\s*B\b", "B.A. LL.B."),
    (r"\bB\s*B\s*A\s*LL\s*\.\s*B\b", "BBA LL.B."),
    (r"\bB\s*\.?\s*Tech\s*\.?\b|\bBachelor of Technology\b", "B.Tech"),
    (r"\bB\s*\.?\s*E\s*\.?\b|\bBachelor of Engineering\b", "B.E."),
    (r"\bM\s*\.?\s*Tech\s*\.?\b|\bMaster of Technology\b", "M.Tech"),
    (r"\bM\s*\.?\s*E\s*\.?\b|\bMaster of Engineering\b", "M.E."),
    (r"\bM\s*\.?\s*C\s*\.\s*A\s*\.?\b|\bMaster of Computer Applications\b", "MCA"),
    (r"\bB\s*\.?\s*C\s*\.\s*A\s*\.?\b|\bBachelor of Computer Applications\b", "BCA"),
    (r"\bB\s*\.?\s*Com\s*\.?\b|\bBachelor of Commerce\b", "B.Com"),
    (r"\bM\s*\.?\s*B\s*\.?\s*A\s*\.?\b|\bMaster of Business Administration\b", "MBA"),
    (r"\bM\s*\.?\s*Com\s*\.?\b|\bMaster of Commerce\b", "M.Com"),
    (r"\bLL\s*\.\s*B\s*\.?\b|\bBachelor of Laws\b", "LLB"),
    (r"\bLL\s*\.\s*M\s*\.?\b|\bMaster of Laws\b", "LLM"),
    (r"\bM\s*\.?\s*Sc\s*\.?\b|\bMaster of Science\b", "M.Sc"),
    (r"\bB\s*\.?\s*Sc\s*\.?\b|\bBachelor of Science\b", "B.Sc"),
    (r"\bB\s*\.\s*A\s*\.?\b|\bBachelor of Arts\b", "B.A."),
    (r"\bM\s*\.\s*A\s*\.?\b|\bMaster of Arts\b", "M.A."),
    (r"\bBFA\b|\bBachelor of Fine Arts\b", "BFA"),
    (r"\bPh\s*\.?\s*D\s*\.?\b", "Ph.D"),
    (r"\bMMS\b|\bMaster of Management Studies\b", "MMS"),
    (r"\bBachelor(?:'s)? in\b", "Bachelor's Degree"),
]
DEGREE_RANK = {
    "Ph.D": 6,
    "M.Tech": 5, "M.E.": 5, "MCA": 5, "MBA": 5, "M.Com": 5,
    "LLM": 5, "M.Sc": 5, "M.A.": 5, "MMS": 5,
    "B.Tech": 4, "B.E.": 4, "BCA": 4, "B.Com": 4, "LLB": 4,
    "B.Sc": 4, "B.A.": 4, "BFA": 4, "B.A. LL.B.": 4, "BBA LL.B.": 4,
    "Bachelor's Degree": 4,
}
COURSE_ALIASES = {
    "cse": "Computer Science and Engineering",
    "computer science": "Computer Science",
    "it": "Information Technology",
    "extc": "Electronics and Telecommunications",
    "electronics and telecommunication": "Electronics and Telecommunications",
}
SKILL_ALIASES = {"js": "JavaScript", "py": "Python", "ml": "Machine Learning"}


def present(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip().lower() not in {"", "nan", "none", "unknown", "[]"}


def value_or_na(value: Any) -> str:
    return str(value).strip() if present(value) else NA


def extract_age(text: str) -> str:
    m = re.search(r"\bage\s*[:=-]\s*(\d{1,2})\b", text or "", re.I)
    if not m:
        return NA
    age = int(m.group(1))
    return str(age) if 16 <= age <= 100 else NA


def extract_degree(education: str) -> str:
    candidates = degree_matches(education)
    if not candidates:
        return NA
    return max(candidates, key=lambda item: (DEGREE_RANK.get(item[1], 0), -item[0].start()))[1]


def degree_matches(education: str) -> list[tuple[re.Match[str], str]]:
    text = education or ""
    matches = []
    occupied: list[tuple[int, int]] = []
    for pattern, canonical in DEGREES:
        for match in re.finditer(pattern, text, re.I):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            matches.append((match, canonical))
            occupied.append((match.start(), match.end()))
    return sorted(matches, key=lambda item: item[0].start())


def extract_degrees(education: str) -> list[str]:
    """Return distinct recognized degrees in source order; diploma credentials are excluded."""
    result = []
    for _, canonical in degree_matches(education):
        if canonical not in result:
            result.append(canonical)
    return result


COURSE_PATTERNS = [
    (r"computer science\s*(?:and|&)\s*engineering|\bCSE\b", "Computer Science and Engineering"),
    (r"computer science", "Computer Science"),
    (r"information technology", "Information Technology"),
    (r"electronics\s*(?:and|&)\s*telecommunications?|\bEXTC\b", "Electronics and Telecommunications"),
    (r"electronics\s*(?:and|&)\s*communication", "Electronics and Communication"),
    (r"electrical\s*(?:and|&)\s*electronics engineering", "Electrical and Electronics Engineering"),
    (r"electrical engineering", "Electrical Engineering"),
    (r"mechanical engineering", "Mechanical Engineering"),
    (r"civil engineering", "Civil Engineering"),
    (r"computer engineering", "Computer Engineering"),
    (r"data science", "Data Science"), (r"computer applications", "Computer Applications"),
    (r"business administration", "Business Administration"), (r"hospitality management", "Hospitality Management"),
    (r"hotel management", "Hotel Management"), (r"marketing(?:\s+and\s+sales)?", "Marketing"),
    (r"operations", "Operations"), (r"finance", "Finance"), (r"human resources", "Human Resources"),
    (r"\blaw\b", "Law"), (r"fine arts", "Fine Arts"), (r"graphic design", "Graphic Design"),
    (r"visual arts", "Visual Arts"), (r"commerce", "Commerce"),
    (r"\bmanagement\b", "Management"),
]


def course_for_degree(education: str, degree_match: re.Match[str] | None) -> tuple[str, str]:
    """Extract course terms only from the selected credential's local text span."""
    text = education or ""
    if degree_match:
        start = degree_match.end()
        # Bound a credential at the next credential or a strong institution/location boundary.
        following = [m.start() for m, _ in degree_matches(text) if m.start() > start]
        end = min(following) if following else min(len(text), start + 180)
        section = text[start:end]
        section = re.split(r"\b(?:University|College|Institute|Institution)\b|,\s*(?:Mumbai|Pune|Chennai|Delhi|Kolkata|Hyderabad|Jaipur|Bangalore|Bengaluru|Guwahati|Indore|Nashik|Nagpur|Maharashtra|Tamil Nadu|Punjab|Rajasthan|Karnataka|Andhra Pradesh)\b", section, maxsplit=1, flags=re.I)[0]
    else:
        section = text
    labeled = re.search(r"\b(?:speciali[sz]ation|course|major|branch)\s*[:=-]\s*([^,;\n]+)", section, re.I)
    if labeled:
        raw = labeled.group(1).strip()
        if re.search(r"\b(?:manager|engineer|developer|consultant|analyst|lead|administrator)\b", raw, re.I):
            return NA, NA
        return COURSE_ALIASES.get(raw.lower(), raw), raw
    for pattern, canonical in COURSE_PATTERNS:
        m = re.search(pattern, section, re.I)
        if m:
            raw = m.group(0).strip()
            return canonical, raw
    return NA, NA


def extract_course(education: str, selected_degree: str | None = None) -> tuple[str, str]:
    """Return (canonical course, raw course) associated with the selected degree."""
    matches = degree_matches(education or "")
    if matches:
        target = selected_degree or extract_degree(education)
        candidates = [item for item in matches if item[1] == target]
        for candidate in candidates:
            course = course_for_degree(education, candidate[0])
            if course[0] != NA:
                return course
        if candidates:
            return course_for_degree(education, candidates[0][0])
    text = education or ""
    # Without a recognized credential, accept only course-like wording preceding any occupation terms.
    course_text = re.split(r"\b(?:Operations Manager|Project Manager|Software Developer|Engineer|Consultant|Analyst|Administrator)\b", text, maxsplit=1, flags=re.I)[0]
    m = re.search(r"\b(?:speciali[sz]ation|course|major|branch)\s*[:=-]\s*([^,;\n]+)", course_text, re.I)
    if m:
        raw = m.group(1).strip()
        return COURSE_ALIASES.get(raw.lower(), raw), raw
    return NA, NA

# Skill, employment, and project extraction.
def extract_skills(raw: str) -> list[str]:
    if not present(raw):
        return []
    cleaned_raw = re.split(r"\b(?:company\s+-|education\s+details|project\s+details|responsibilities)", str(raw), maxsplit=1, flags=re.I)[0]
    cleaned_raw = re.sub(r"\b(?:web development|database|development tools|framework|server|operating systems|key skills|technical skills|skills)\s*:\s*", "", cleaned_raw, flags=re.I)
    cleaned_raw = re.sub(r"(?<=[A-Za-z0-9])\.\s+(?=[A-Z])", ", ", cleaned_raw)
    marker = re.compile(r"(?:Exprience|Experience)\s*[-:]\s*(?:\d+(?:\.\d+)?|less than\s+\d+(?:\.\d+)?)\s*(?:months?|years?)(?:\s+months?)?", re.I)
    labeled, cursor = [], 0
    for match in marker.finditer(cleaned_raw):
        skill = re.sub(r"\s*[-:]+\s*$", "", cleaned_raw[cursor:match.start()]).strip()
        if skill:
            labeled.append(skill)
        cursor = match.end()
    vals = labeled if labeled else re.split(r"\s*(?:[,;|\n]+)\s*", cleaned_raw)
    result = []
    for val in vals:
        val = re.sub(r"\s*[-–]?\s*(?:Exprience|Experience)\s*[-:]?.*$", "", val, flags=re.I).strip(" -:")
        val = ".NET" if val.strip().casefold() == ".net" else val.strip(" .")
        val = re.sub(r"^(?:web development|database|development tools|framework|server|operating systems|key skills|technical skills|skills)\s*:\s*", "", val, flags=re.I).strip()
        if val.casefold() in {"employee resource group", "engineer", "skills", "skill details", "technical skills", "key skills"}:
            continue
        if re.search(r"\b(?:engineer|manager|developer|consultant|administrator|analyst)\s*$", val, re.I):
            continue
        if not val or re.fullmatch(r"(?:months?|years?)\b", val, re.I):
            continue
        norm = SKILL_ALIASES.get(val.lower(), val)
        if norm.casefold() not in {x.casefold() for x in result}:
            result.append(norm)
    return result


MONTHS = {name.lower()[:3]: i for i, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1
)}
MONTHS["sep"] = 9


def extract_experience_years(text: str, as_of: date | None = None) -> str:
    """Extract explicit employment periods, union overlaps, and reject project/skill durations."""
    if not present(text):
        return NA
    source = str(text)
    as_of = as_of or date.today()
    month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    atom = rf"(?:{month}\.?\s+\d{{4}}|\d{{4}})"
    range_pattern = re.compile(rf"(?P<start>{atom})\s*(?:to|through|[-\u2013])\s*(?P<end>{atom}|present|current|till\s+date|to\s+date)", re.I)

    def parse_month(value: str, end: bool = False) -> int | None:
        if re.search(r"present|current|till\s+date|to\s+date", value, re.I):
            return as_of.year * 12 + as_of.month - 1
        m = re.fullmatch(rf"({month})\.?\s+(\d{{4}})", value.strip(), re.I)
        if m:
            return int(m.group(2)) * 12 + MONTHS[m.group(1).lower()[:3]] - 1
        m = re.fullmatch(r"(\d{4})", value.strip())
        return int(m.group(1)) * 12 + (11 if end else 0) if m else None

    intervals = []
    for match in range_pattern.finditer(source):
        context = source[max(0, match.start() - 140):min(len(source), match.end() + 100)]
        if re.search(r"\b(?:project|skill|duration\s*:|client/project)\b", context, re.I):
            continue
        if not re.search(r"\b(?:worked|working|employed|joined|employment|currently|position|designation|as\s+(?:a|an))\b", context, re.I):
            continue
        start, end = parse_month(match.group("start")), parse_month(match.group("end"), True)
        if start is not None and end is not None and start <= end and end - start < 80 * 12:
            intervals.append((start, end))
    if intervals:
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        months_total = sum(end - start for start, end in merged)
        return str(round(months_total / 12, 2)) if months_total <= 80 * 12 else NA

    total = re.search(r"\btotal\s+(?:professional\s+)?experience\s*[:=-]\s*(\d+(?:\.\d+)?)\s*(years?|months?)\b", source, re.I)
    if not total:
        total = re.search(r"\b(\d+(?:\.\d+)?)\s*(years?|months?)\s+of\s+(?:total\s+)?work\s+experience\b", source, re.I)
    if total:
        months_total = float(total.group(1)) * (12 if total.group(2).lower().startswith("year") else 1)
        return str(round(months_total / 12, 2)) if months_total <= 80 * 12 else NA

    durations = []
    for match in re.finditer(r"\b(?:for\s+)?(\d+(?:\.\d+)?)\s*(years?|months?)\b", source, re.I):
        context = source[max(0, match.start() - 100):min(len(source), match.end() + 50)]
        if re.search(r"\b(?:project|skill|duration\s*:|experience\s*-\s*\d)\b", context, re.I):
            continue
        if re.search(r"\b(?:worked|working|employed|employment|position|designation|as\s+(?:a|an))\b", context, re.I):
            durations.append(float(match.group(1)) * (12 if match.group(2).lower().startswith("year") else 1))
    # Undated multiple job durations may overlap, so leave their total unknown.
    role_mentions = len(re.findall(r"\b(?:worked|working|employed|joined|currently working|as\s+(?:a|an))\b", source, re.I))
    if len(durations) == 1 and role_mentions <= 1 and durations[0] <= 80 * 12:
        return str(round(durations[0] / 12, 2))
    return NA


def extract_projects(text: str) -> list[str]:
    """Extract explicitly labeled project names/titles without swallowing descriptions."""
    if not present(text):
        return []
    header = re.compile(r"(?<![A-Za-z/])(?:Project[- ](?:I{1,3}|IV|[1-9])|(?:(?:(?:B\s*\.?\s*E\.?|M\s*\.?\s*E\.?|Diploma|Final\s+Year|Mini|Academic|Major)\s+)?Project(?:\s+(?:Name|Title|Profile))?))\s*[:=-]\s*", re.I)
    stop = re.compile(r"\b(?:technolog(?:y|ies)\s+used|tools?(?:\s+and\s+technologies)?|about\s+project|project\s+type|responsibilities|role|synopsis|team\s+size|position|education\s+details|skill\s+details|company\s+details|client)\s*[:=-]", re.I)
    titles, source = [], str(text)
    matches = list(header.finditer(source))
    for i, match in enumerate(matches):
        if re.search(r"\babout\s+$", source[max(0, match.start() - 15):match.start()], re.I):
            continue
        if re.search(r"\bin\s+$", source[max(0, match.start() - 30):match.start()], re.I):
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        block = source[match.end():end]
        stop_match = stop.search(block)
        if stop_match:
            block = block[:stop_match.start()]
        block = re.split(r"\s+Project\s+(?:\u00e2\u0080\u00a2|:)|\u00e2\u0080\u00a2|\u00e2\u009d\u0096|\u00e2\u009e\u00a2", block, maxsplit=1)[0]
        dash = re.search(r"\s+-\s+(?:It's\b|[A-Z][A-Za-z0-9&.']+\s+(?:is|was|were|has|had|aims|supports|provides)\b)", block)
        if dash:
            block = block[:dash.start()]
        block = re.split(r"[.;\n•]+", block, maxsplit=1)[0]
        title = re.sub(r"\s+", " ", block).strip(" .;:-•")
        title = title.lstrip("* ").strip()
        title = re.sub(r"^(?:DIPLOMA|DEGREE)\s*:\s*", "", title, flags=re.I)
        title = re.sub(r"^Project\s+based\s+on\s+", "", title, flags=re.I)
        title = re.sub(r"^Topic\s*:\s*", "", title, flags=re.I)
        title = re.sub(r"^[IVX]+\s*:\s*", "", title, flags=re.I)
        if title.casefold() in {"diploma", "degree"}:
            continue
        title = re.split(r"\)\s+[A-Z][A-Za-z0-9&.'() -]{0,25}\s+(?:is|was|were|has|had|aims|supports|provides)\b", title, maxsplit=1)[0].strip()
        title = re.split(r"\s+Project\s+(?:\u00e2\u0080\u00a2|is\b)", title, maxsplit=1, flags=re.I)[0].strip()
        if re.match(r"^(?:Project\s+)?(?:Status|Type|Details)\b", title, re.I):
            continue
        if title and title.casefold() not in {item.casefold() for item in titles}:
            titles.append(title)
    return titles


def structure_resume(row: dict[str, Any]) -> dict[str, Any]:
    text = value_or_na(row.get("resume_text"))
    education = value_or_na(row.get("education_raw"))
    degree = extract_degree(education)
    course, course_raw = extract_course(education)
    detected_degrees = extract_degrees(education)
    skills = extract_skills(row.get("skills_raw", ""))
    return {
        "resume_id": value_or_na(row.get("resume_id")), "name": value_or_na(row.get("name")),
        "age": extract_age(text), "college": value_or_na(row.get("college")),
        "college_tier": value_or_na(row.get("college_tier")), "city": value_or_na(row.get("city")),
        "city_tier": value_or_na(row.get("city_tier")), "degree": degree,
        "detected_degrees": "|".join(detected_degrees) if detected_degrees else NA,
        "course": course, "specialization": course, "course_raw": course_raw,
        "institution_extracted": NA, "education_year": NA,
        "total_experience_years": extract_experience_years(row.get("experience_raw", "")),
        "employment_gap_months": value_or_na(row.get("employment_gap_months")),
        "skills": "|".join(skills) if skills else NA,
        "projects": "|".join(extract_projects(text)) or NA,
        "source_category": value_or_na(row.get("source_category")), "resume_text": text,
        "education_raw": education, "skills_raw": value_or_na(row.get("skills_raw")),
        "experience_raw": value_or_na(row.get("experience_raw")),
        # Compatibility aliases used by run_screening.py; values remain evidence-based.
        "primary_degree": "" if degree == NA else degree, "course_final": "" if course == NA else course,
        "skills_clean": "|".join(skills),
        "skills_extracted": "|".join(skills),
        "skills_clean_new": "|".join(skills),
        "projects_extracted": "|".join(extract_projects(text)),
        "total_experience_years_final": "" if extract_experience_years(row.get("experience_raw", "")) == NA else extract_experience_years(row.get("experience_raw", "")),
        "age_final": "" if extract_age(text) == NA else extract_age(text), "experience_source": "explicit_text" if extract_experience_years(row.get("experience_raw", "")) != NA else NA,
        "age_source": "explicit_text" if extract_age(text) != NA else NA,
    }


def standardize_jobs(jobs: pd.DataFrame, mappings: pd.DataFrame | None = None) -> pd.DataFrame:
    company = pd.Series([NA] * len(jobs), index=jobs.index)
    if mappings is not None and {"job_id", "company"}.issubset(mappings.columns):
        company_map = mappings.drop_duplicates("job_id").set_index("job_id")["company"]
        company = jobs["job_id"].map(company_map).fillna(NA)
    job_text = jobs.apply(lambda r: "; ".join(
        f"{label}: {r[col]}" for label, col in [
            ("Role", "job_title"), ("Preferred degree", "preferred_degree"),
            ("Accepted degrees", "accepted_degrees"), ("Preferred course", "preferred_course"),
            ("Accepted courses", "accepted_courses"), ("Minimum experience years", "minimum_total_experience_years"),
            ("Required skills", "required_skills"),
        ] if col in r.index and present(r[col])
    ), axis=1)
    out = pd.DataFrame({
        "job_id": jobs.get("job_id", pd.Series(dtype=str)), "company": company,
        "job_title": jobs.get("job_title", NA), "required_degree": NA,
        "preferred_degree": jobs.get("preferred_degree", NA),
        "accepted_degrees": jobs.get("accepted_degrees", NA),
        "required_courses": NA, "accepted_courses": jobs.get("accepted_courses", NA),
        "preferred_courses": jobs.get("preferred_course", NA),
        "minimum_experience": jobs.get("minimum_total_experience_years", NA), "preferred_experience": NA,
        "required_skills": jobs.get("required_skills", NA), "preferred_skills": NA,
        "job_description": jobs.get("job_description", job_text),
    })
    if "job_description" not in jobs:
        out["job_description"] = job_text
    return out


def transform_counterfactual(original: dict[str, Any], cf_id: str, attribute: str, value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed = {"name", "age", "college", "college_tier", "city", "city_tier", "degree", "course", "specialization", "total_experience_years", "employment_gap_months"}
    if attribute not in allowed:
        raise ValueError(f"Unsupported counterfactual attribute: {attribute}")
    variant = dict(original)
    old = value_or_na(original.get(attribute))
    new = value_or_na(value)
    variant[attribute] = new
    record = {"original_resume_id": original.get("resume_id", NA), "counterfactual_id": cf_id,
              "changed_attribute": attribute, "original_value": old, "counterfactual_value": new}
    return variant, record


def structure_existing_counterfactuals(resumes: pd.DataFrame, legacy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain only legacy variants whose stated before-values match source-derived fields."""
    lookup = resumes.set_index("resume_id").to_dict("index")
    accepted, review = [], []
    for _, row in legacy.iterrows():
        rid = str(row.get("original_resume_id", ""))
        attrs = str(row.get("changed_attribute", "")).split("+")
        olds = str(row.get("original_value", "")).split("|") if len(attrs) > 1 else [str(row.get("original_value", ""))]
        news = str(row.get("counterfactual_value", "")).split("|") if len(attrs) > 1 else [str(row.get("counterfactual_value", ""))]
        original = lookup.get(rid)
        reason = ""
        if original is None:
            reason = "parent resume is missing"
        elif len(olds) != len(attrs) or len(news) != len(attrs):
            reason = "combined-attribute values do not align"
        else:
            for attr, old in zip(attrs, olds):
                field = "total_experience_years" if attr == "experience" else attr
                if field not in original:
                    reason = f"unsupported attribute: {attr}"; break
                if value_or_na(original[field]).casefold() != value_or_na(old).casefold():
                    reason = f"source value mismatch for {attr}: structured={original[field]}, legacy={old}"; break
        if reason:
            review.append({"check": "legacy_counterfactual_review", "record_id": row.get("cf_id", NA), "detail": reason})
            continue
        variant = dict(original)
        for attr, value in zip(attrs, news):
            field = "total_experience_years" if attr == "experience" else attr
            variant[field] = value_or_na(value)
        changed = [key for key in original if key != "resume_id" and value_or_na(original[key]) != value_or_na(variant.get(key))]
        if set(changed) != set(attrs):
            review.append({"check": "legacy_counterfactual_review", "record_id": row.get("cf_id", NA), "detail": f"intended changes {attrs}; actual changes {changed}"})
            continue
        output = {"original_resume_id": rid, "counterfactual_id": row.get("cf_id", NA), "cf_id": row.get("cf_id", NA),
                         "changed_attribute": row.get("changed_attribute", NA), "original_value": row.get("original_value", NA),
                         "counterfactual_value": row.get("counterfactual_value", NA), **variant,
                         "variant_json": json.dumps(variant, ensure_ascii=False)}
        accepted.append(output)
    cf_columns = ["original_resume_id", "counterfactual_id", "cf_id", "changed_attribute", "original_value", "counterfactual_value", *resumes.columns, "variant_json"]
    return pd.DataFrame(accepted, columns=cf_columns), pd.DataFrame(review, columns=["check", "record_id", "detail"])


def validate_data(resumes: pd.DataFrame, counterfactuals: pd.DataFrame | None = None, mappings: pd.DataFrame | None = None, jobs: pd.DataFrame | None = None) -> pd.DataFrame:
    issues: list[dict[str, str]] = []
    def issue(check: str, rid: Any, detail: str) -> None:
        issues.append({"check": check, "record_id": str(rid), "detail": detail})
    required = ["resume_id", "name", "college", "degree", "course", "resume_text"]
    for col in required:
        if col not in resumes:
            issue("missing_required_column", "NA", col)
        else:
            for ix in resumes.index[resumes[col].isna() | (resumes[col].astype(str).str.strip() == "")]:
                issue("missing_required_value", resumes.at[ix, "resume_id"] if "resume_id" in resumes else ix, col)
    if "resume_id" in resumes:
        for rid in resumes.loc[resumes.resume_id.duplicated(keep=False), "resume_id"]: issue("duplicate_resume_id", rid, "resume_id must be unique")
    valid_degrees = {x[1] for x in DEGREES} | {NA}
    if "degree" in resumes:
        for _, row in resumes[~resumes.degree.isin(valid_degrees)].iterrows(): issue("invalid_degree", row.get("resume_id"), row.get("degree"))
    if {"degree", "course"}.issubset(resumes.columns):
        for _, row in resumes.iterrows():
            course = value_or_na(row.get("course"))
            degree = value_or_na(row.get("degree"))
            if course != NA and (len(course) > 120 or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", course)):
                issue("malformed_course", row.get("resume_id"), course)
            if course != NA and degree != NA and course.casefold() == degree.casefold():
                issue("inconsistent_degree_course", row.get("resume_id"), f"degree and course both equal {course}")
    if "total_experience_years" in resumes:
        nums = pd.to_numeric(resumes.total_experience_years.replace(NA, pd.NA), errors="coerce")
        for ix in resumes.index[(nums < 0) | (nums > 80)]: issue("impossible_experience", resumes.at[ix, "resume_id"], str(nums[ix]))
    if counterfactuals is not None and len(counterfactuals):
        idcol = "counterfactual_id" if "counterfactual_id" in counterfactuals else "cf_id"
        for cfid in counterfactuals.loc[counterfactuals[idcol].duplicated(keep=False), idcol]: issue("duplicate_counterfactual_id", cfid, "counterfactual ID must be unique")
        value_cols = [c for c in resumes.columns if c != "resume_id"]
        original_lookup = resumes.set_index("resume_id").to_dict("index")
        for _, cf in counterfactuals.iterrows():
            rid = cf.get("original_resume_id")
            if rid not in original_lookup: issue("unknown_counterfactual_parent", rid, str(cf.get(idcol))); continue
            declared = str(cf.get("changed_attribute", "")).split("+")
            allowed_attrs = {"total_experience_years" if a == "experience" else a for a in declared}
            actual_changes = set()
            for col in value_cols:
                baseval = value_or_na(original_lookup[rid].get(col))
                cfval = value_or_na(cf.get(col)) if col in cf.index else baseval
                if cfval != baseval:
                    actual_changes.add(col)
                    if col not in allowed_attrs:
                        issue("unintended_counterfactual_change", cf.get(idcol), f"{col}: {baseval} -> {cfval}")
            if actual_changes != allowed_attrs:
                issue("counterfactual_change_mismatch", cf.get(idcol), f"declared={sorted(allowed_attrs)} actual={sorted(actual_changes)}")
    if jobs is not None and mappings is not None:
        if "job_id" in jobs and "job_id" in mappings:
            for jid in set(jobs.job_id.dropna()) - set(mappings.job_id.dropna()): issue("missing_company_course_mapping", jid, "no mapping row")
    if not issues: issues.append({"check": "passed", "record_id": NA, "detail": "No validation errors found."})
    return pd.DataFrame(issues, columns=["check", "record_id", "detail"])


def build() -> None:
    source = pd.read_csv(ROOT / "data" / "parsed_resumes.csv", keep_default_na=False)
    resumes = pd.DataFrame([structure_resume(row) for row in source.to_dict("records")])
    resumes.to_csv(ROOT / "data" / "structured_resumes.csv", index=False)
    jobs = pd.read_csv(ROOT / "job_requirements_complete.csv", keep_default_na=False)
    # Preserve project mapping and make uncertainty explicit. Existing mappings cite their project-defined JD source.
    mappings = pd.read_csv(ROOT / "final" / "company_course_preferences.csv", keep_default_na=False)
    standardize_jobs(jobs, mappings).to_csv(ROOT / "data" / "structured_job_descriptions.csv", index=False)
    mappings["unrelated_courses"] = NA
    mappings["degree_preferences_separate"] = "See job_requirements_complete.csv"
    mappings["mapping_status"] = "project_defined"
    mappings.to_csv(ROOT / "data" / "company_course_preference_mapping.csv", index=False)
    legacy = pd.read_csv(ROOT / "final" / "counterfactual_resumes_valid.csv", keep_default_na=False)
    structured_cf, review = structure_existing_counterfactuals(resumes, legacy)
    structured_cf.to_csv(ROOT / "data" / "structured_counterfactuals.csv", index=False)
    report = validate_data(resumes, counterfactuals=structured_cf, mappings=mappings, jobs=jobs)
    for generated in ["structured_resumes.csv", "structured_job_descriptions.csv", "company_course_preference_mapping.csv", "structured_counterfactuals.csv"]:
        pd.read_csv(ROOT / "data" / generated, keep_default_na=False, on_bad_lines="error")
    report = pd.concat([report, pd.DataFrame([{"check": "malformed_csv", "record_id": NA, "detail": "Generated CSVs parse successfully."}])], ignore_index=True)
    pd.concat([report, review], ignore_index=True).to_csv(ROOT / "validation" / "structured_data_validation.csv", index=False)


if __name__ == "__main__":
    build()
