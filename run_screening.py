
import os, re, ast, argparse, json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from screening.requirements import METHOD_VERSION, UPLOAD_METHOD_VERSION, classify_jd_requirements, requirement_skill_names

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Final score = 100%
WEIGHTS = {
    "skills": 0.30,
    "experience": 0.20,
    "education_course": 0.15,
    "projects": 0.15,
    "semantic": 0.20,
}

def clean(x):
    return "" if pd.isna(x) else str(x).strip()

def norm(x):
    x = clean(x).lower().replace("&", " and ")
    x = re.sub(r"[\u2010-\u2015]", "-", x)
    x = re.sub(r"[^a-z0-9+#./ -]", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def num(x, default=0.0):
    try:
        if pd.isna(x): return default
        return float(x)
    except Exception:
        return default

def parse_list(x):
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x if clean(v)]
    raw = clean(x)
    if not raw or raw in {"[]", "None", "nan"}:
        return []
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, (list, tuple, set)):
            return [clean(v) for v in obj if clean(v)]
    except Exception:
        pass
    raw = raw.strip("[]")
    parts = re.split(r"\s*\|\s*|\s*;\s*|\s*,\s*", raw)
    return [p.strip(" '\"") for p in parts if p.strip(" '\"")]

def norm_skill(x):
    x = norm(x)
    x = re.sub(r"\s*-\s*(?:exprience|experience).*?$", "", x)
    x = re.sub(r"\s+\d+(?:\.\d+)?\s*months?\b", "", x)
    return re.sub(r"\s+", " ", x).strip()

def resume_skills(row):
    for col in ["skills_clean_new", "skills_extracted", "skills_clean"]:
        vals = parse_list(row.get(col, ""))
        if vals:
            return {norm_skill(v) for v in vals if norm_skill(v)}
    return set()

def required_skills(job):
    return {norm_skill(x) for x in requirement_skill_names(job) if norm_skill(x)}

ALIASES = {
    "aws": {"aws", "amazon web services"},
    "amazon web services": {"aws", "amazon web services"},
    "sql": {"sql", "mysql"},
    "mysql": {"mysql", "sql"},
    "databases": {"database", "databases", "mysql", "oracle"},
    "database": {"database", "databases", "mysql", "oracle"},
    "javascript": {"javascript", "javascript/typescript"},
    "html": {"html", "html5"},
    "css": {"css", "css3"},
    "networking": {"networking", "computer networking"},
    "automation": {"automation", "test automation", "automation testing"},
}

def skill_match(req, have):
    return req == have or have in ALIASES.get(req, set())

def _resume_evidence_location(text, excerpt):
    """Describe the section/type for a literal resume excerpt without inference."""
    if not excerpt:
        return "Not demonstrated", "none"
    offset = str(text).find(excerpt)
    preceding = str(text)[:offset] if offset >= 0 else ""
    headings = {
        "skills": "Skills", "technical skills": "Skills", "key skills": "Skills",
        "projects": "Projects", "project": "Projects", "experience": "Experience",
        "work experience": "Experience", "employment": "Experience", "internships": "Internships",
        "internship": "Internships", "certifications": "Certifications", "education": "Education",
    }
    section = "Resume text"
    for line in preceding.splitlines():
        normalized = re.sub(r"[^a-z ]", "", line.casefold()).strip()
        if normalized in headings:
            section = headings[normalized]
    if section == "Skills": evidence_type = "skills_list_mention"
    elif section == "Projects": evidence_type = "project"
    elif section == "Internships" or re.search(r"\bintern\b", preceding[-800:], re.I): evidence_type = "internship"
    elif section == "Experience": evidence_type = "professional_experience"
    else: evidence_type = "resume_text_mention"
    return section, evidence_type

def score_skills(row, job, category_weights=None):
    req = required_skills(job)
    if not req: return None
    have = resume_skills(row)
    text = clean(row.get("resume_text", ""))
    req_context = job.get("requirements", {})
    if isinstance(req_context, str):
        try: req_context = json.loads(req_context)
        except (ValueError, TypeError): req_context = {}
    req_records = req_context.get("requirements", []) if isinstance(req_context, dict) else []
    source_by_skill = {}
    category_priority = {"explicit_minimum": 4, "preferred": 3, "uncertain": 2, "general_mention": 1}
    for item in req_records:
        if item.get("kind") in {"skill", "experience_duration"} and item.get("criterion") != "total professional experience":
            key = norm_skill(item.get("criterion", ""))
            current = source_by_skill.get(key, {})
            item_priority = category_priority.get(item.get("category"), 0)
            current_priority = category_priority.get(current.get("category"), 0)
            if (key not in source_by_skill or item_priority > current_priority
                    or (item_priority == current_priority and item.get("duration_years") is not None
                        and current.get("duration_years") is None)):
                source_by_skill[key] = item
    requirement_evidence = []
    weighted_total = 0.0
    criterion_weight_total = 0.0
    category_weights = category_weights or {}
    for requirement in sorted(req):
        matched = next((skill for skill in sorted(have) if skill_match(requirement, skill)), None)
        excerpt = ""
        evidence_class = "not_demonstrated"
        value = 0.0
        if matched:
            is_exact = norm_skill(requirement) == norm_skill(matched)
            evidence_class = "listed_mention" if is_exact else "related_transferable_skill"
            value = 0.5
            # Upgrade only when the matched term occurs in an explicit work/action
            # statement; a list alone is not evidence of professional use.
            pat = re.compile(r"(?<![\w])" + re.escape(matched) + r"(?![\w])", re.I)
            for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
                if pat.search(sentence) and re.search(r"\b(?:used|using|developed|built|deployed|implemented|tested|automated|maintained|worked|created|configured|supported)\b", sentence, re.I):
                    excerpt = sentence.strip()
                    evidence_class = "professional_use_evidenced" if is_exact else "related_skill_use_evidenced"
                    value = 1.0
                    break
            if not excerpt:
                excerpt = next((sentence.strip() for sentence in re.split(r"\n|[.;]", text) if pat.search(sentence)), "")
        jd_requirement = source_by_skill.get(requirement, {})
        category = jd_requirement.get("category", "explicit_structured_requirement")
        criterion_weight = float(category_weights.get(category, 1.0))
        weighted_total += value * criterion_weight
        criterion_weight_total += criterion_weight
        section, evidence_type = _resume_evidence_location(text, excerpt)
        requirement_evidence.append({"requirement": requirement, "evidence_class": evidence_class,
                                     "evidence_status": ("not_demonstrated" if not matched else
                                                         "demonstrated" if evidence_class == "professional_use_evidenced" else
                                                         "partially_demonstrated" if evidence_class.startswith("related_") else "listed"),
                                     "match_status": "not_demonstrated" if not matched else "partially_matched" if evidence_class.startswith("related_") else "matched",
                                     "resume_skill": matched, "resume_excerpt": excerpt,
                                     "resume_section": section, "evidence_type": evidence_type,
                                     "criterion_weight": criterion_weight,
                                     "weighted_evidence_credit": round(value * criterion_weight, 6),
                                     "source": "resume_text_or_structured_skill" if matched else "not found in supplied resume",
                                     "jd_requirement_category": jd_requirement.get("category", "explicit_structured_requirement"),
                                     "jd_source_text": jd_requirement.get("source_text", "Explicit structured JD skills field"),
                                     "jd_duration_years": jd_requirement.get("duration_years"),
                                         "jd_proficiency_requirement": (jd_requirement.get("source_text") if re.search(
                                             r"\b(?:proficien|expert|advanced|beginner|intermediate)\b",
                                             str(jd_requirement.get("source_text", "")), re.I) else next(
                                             (str(record.get("source_text", "")) for record in req_records
                                              if record.get("kind") == "statement" and re.search(r"\b(?:proficien|expert|advanced|beginner|intermediate)\b", str(record.get("source_text", "")), re.I)), None))})
    row["_skill_evidence"] = requirement_evidence
    row["_skill_score_calculation"] = {
        "weighted_evidence_total": round(weighted_total, 6),
        "criterion_weight_total": round(criterion_weight_total, 6),
        "score": round(weighted_total / criterion_weight_total, 6) if criterion_weight_total else None,
        "category_weights": dict(category_weights),
        "policy": "Weighted mean of per-skill evidence credit by JD requirement category; exact/professional evidence credit is unchanged.",
    }
    return weighted_total / criterion_weight_total if criterion_weight_total else None

def experience_years(row):
    y = num(row.get("total_experience_years_final", np.nan), np.nan)
    if not np.isnan(y): return max(0.0, y)
    m = num(row.get("total_experience_months_final", np.nan), np.nan)
    if not np.isnan(m): return max(0.0, m/12.0)
    return max(0.0, num(row.get("total_experience_years", 0)))

def score_experience(row, job):
    parsed = job.get("requirements", [])
    if isinstance(parsed, str):
        try: parsed = json.loads(parsed)
        except (ValueError, TypeError): parsed = []
    if isinstance(parsed, dict): parsed = parsed.get("requirements", [])
    records = row.get("experience_records", [])
    if isinstance(records, str):
        try: records = json.loads(records)
        except (ValueError, TypeError): records = []
    category_months = row.get("experience_months_by_category", {})
    if isinstance(category_months, str):
        try: category_months = json.loads(category_months)
        except (ValueError, TypeError): category_months = {}
    total_months = row.get("total_professional_experience_months")
    if pd.isna(total_months): total_months = None
    raw_total_years = row.get("total_experience_years", "")
    total_years = experience_years(row)
    if pd.isna(raw_total_years) or str(raw_total_years).strip().upper() in {"", "NA", "N/A", "NONE"}:
        total_years = None
    common_evidence = {
        "total_professional_years": total_years,
        "total_professional_months": total_months,
        "reported_total_professional_years": row.get("reported_total_experience_years"),
        "experience_source": row.get("experience_source", "unknown"),
        "as_of": row.get("experience_as_of"),
        "experience_records": records or [],
        "combined_months_by_category": category_months or {},
        "duration_convention": row.get("experience_duration_convention", "inclusive_calendar_months"),
        "internship_months": (category_months or {}).get("internship"),
        "uncertain_or_unclassified_months": (category_months or {}).get("unknown_or_requires_review"),
        "required_years": num(job.get("minimum_total_experience_years", 0)),
        "required_months": num(job.get("minimum_total_experience_years", 0)) * 12,
    }
    technology_durations = [item for item in (parsed or []) if item.get("kind") == "experience_duration"
                            and item.get("criterion") != "total professional experience"
                            and item.get("category") in {"explicit_minimum", "preferred"}]
    if technology_durations:
        text = clean(row.get("resume_text", ""))
        evidence_years = []
        evidence = []
        for criterion in technology_durations:
            technology = str(criterion.get("criterion", ""))
            activity = str(criterion.get("activity", ""))
            # Only an explicitly stated duration tied to both the technology and
            # requested activity qualifies. Employment tenure and skills-list
            # mentions do not imply technology-specific years.
            duration_evidence = None
            technology_pattern = re.compile(r"(?<![\w])" + re.escape(technology) + r"(?![\w])", re.I)
            activity_pattern = re.compile(re.escape(activity.rstrip("ing")) + r"\w*", re.I) if activity else None
            for sentence in re.split(r"[.\n;]+", text):
                if not technology_pattern.search(sentence) or (activity_pattern and not activity_pattern.search(sentence)):
                    continue
                duration_match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(years?|yrs?|months?)\b", sentence, re.I)
                if duration_match:
                    duration_evidence = (sentence.strip(), duration_match)
                    break
            if duration_evidence:
                sentence, match = duration_evidence
                amount = float(match.group(1))
                if match.group(2).lower().startswith("month"): amount /= 12
                evidence_years.append(min(amount / float(criterion["duration_years"]), 1.0))
                evidence.append({"technology": technology, "activity": activity, "years": amount, "source_text": sentence})
            else:
                contextual_project = None
                project_headers = list(re.finditer(r"(?im)^\s*Project\s*#?\s*\d+[^\n]*", text))
                for index, header in enumerate(project_headers):
                    end = project_headers[index + 1].start() if index + 1 < len(project_headers) else len(text)
                    block = text[header.start():end]
                    duration_match = re.search(r"(?:~|about|approximately)?\s*(\d+(?:\.\d+)?)\s*(years?|months?)\s+project\b", block, re.I)
                    has_activity = bool(activity_pattern.search(block)) if activity_pattern else False
                    tools_section = re.search(r"(?is)(?:technologies\s*/\s*tools|tools\s+and\s+technologies)\s*(.*?)(?=\bTeam\b|\bOther\b|\Z)", block)
                    technology_in_project_tools = bool(tools_section and technology_pattern.search(tools_section.group(1)))
                    if duration_match and technology_in_project_tools and has_activity:
                        amount = float(duration_match.group(1))
                        if duration_match.group(2).lower().startswith("month"): amount /= 12
                        contextual_project = {"technology": technology, "activity": activity,
                                              "approximate_project_duration_years": amount,
                                              "status": "contextual_project_duration_requires_review",
                                              "source_text": block[:900].strip()}
                        break
                if contextual_project:
                    evidence.append(contextual_project)
                else:
                    evidence.append({"technology": technology, "activity": activity, "years": None,
                                     "status": "insufficient_evidence", "source_text": "No explicit technology/activity duration found."})
        row["_experience_evidence"] = {
            **common_evidence,
            "experience_requirement_type": "technology_specific",
            "note": "Technology-specific scores require explicit resume text tying a duration to the requested technology and activity. Role date ranges are shown separately and do not establish technology years.",
            "technology_specific": evidence,
        }
        return float(sum(evidence_years) / len(evidence_years)) if evidence_years and len(evidence_years) == len(technology_durations) else None
    minimum = num(job.get("minimum_total_experience_years", 0))
    if minimum <= 0:
        row["_experience_evidence"] = {
            **common_evidence,
            "experience_requirement_type": "evidence_only_no_jd_duration",
            "note": "Resume experience evidence is displayed, but the JD provides no numeric experience duration against which to score it.",
            "uncertain_duration_records": [r for r in records if r.get("duration_months") is None],
        }
        return None
    total_requirements = [item for item in (parsed or []) if item.get("kind") == "experience_duration"
                          and item.get("criterion") == "total professional experience"
                          and item.get("category") in {"explicit_minimum", "preferred"}]
    requested_activity = next((str(item.get("activity", "")) for item in total_requirements if item.get("activity")), "")
    activity_requirement_text = " ".join(str(item.get("source_text", "")) for item in total_requirements).casefold()
    relevant_months = None
    relevant_records = []
    if requested_activity:
        eligible_categories = {"full_time_employment", "part_time_employment", "contract_or_freelance", "internship"}
        activity_patterns = {
            "develop": r"\b(?:develop\w*|build\w*|program\w*|software|application|engineer\w*|backend|full[ -]?stack)\b",
            "program": r"\b(?:program\w*|develop\w*|software|application|code|engineer\w*)\b",
            "engineer": r"\b(?:engineer\w*|develop\w*|build\w*|software|application|backend)\b",
            "build": r"\b(?:build\w*|develop\w*|program\w*|software|application|engineer\w*)\b",
        }
        activity_key = next((key for key in activity_patterns if requested_activity.casefold().startswith(key)), "")
        pattern = re.compile(activity_patterns.get(activity_key, rf"\b{re.escape(requested_activity)}\w*\b"), re.I)
        for record in records:
            evidence_text = f"{record.get('role_title', '')} {record.get('supporting_text', '')}"
            domain_supported = True
            if "backend" in activity_requirement_text:
                domain_supported = bool(re.search(r"\b(?:backend|back[ -]?end|server[ -]?side|api|microservice|software\s+(?:engineer|developer|development|application))\b", evidence_text, re.I))
            elif re.search(r"\bsoftware\s+(?:development|engineering)|\bsoftware\s+(?:engineer|developer)\b", activity_requirement_text, re.I):
                role_title = str(record.get("role_title", ""))
                explicit_software_work = re.search(
                    r"\b(?:built|developed|designed|implemented|engineered|programmed|maintained)"
                    r"(?:\W+\w+){0,5}\W+(?:software|application|app|API|service|platform|system|code|microservices?|website)\b",
                    evidence_text, re.I,
                )
                domain_supported = bool(re.search(
                    r"\b(?:software\s+(?:engineer(?:ing)?|developer)|full[ -]?stack|backend|application\s+(?:engineer|developer))\b",
                    role_title, re.I,
                ) or explicit_software_work)
            elif "quality assurance" in activity_requirement_text or re.search(r"\bQA\b", activity_requirement_text):
                domain_supported = bool(re.search(r"\b(?:quality assurance|QA|tester|testing|test automation)\b", evidence_text, re.I))
            if record.get("experience_category") in eligible_categories and pattern.search(evidence_text) and domain_supported:
                relevant_records.append(record)
        intervals = sorted(
            ((int(record["start_year"]) * 12 + int(record["start_month"]) - 1,
              int(record["end_year"]) * 12 + int(record["end_month"]) - 1)
             for record in relevant_records if record.get("start_year") and record.get("start_month")
             and record.get("end_year") and record.get("end_month"))
        )
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        relevant_months = sum(end - start + 1 for start, end in merged) if merged else None
        y = relevant_months / 12 if relevant_months is not None else np.nan
        row["_experience_evidence"] = {
            **common_evidence,
            "experience_requirement_type": "relevant_role",
            "required_activity": requested_activity,
            "activity_requirement_text": activity_requirement_text,
            "relevant_role_months": relevant_months,
            "duration_ratio_numerator_months": relevant_months,
            "duration_ratio_denominator_months": num(job.get("minimum_total_experience_years", 0)) * 12,
            "duration_ratio": (min(relevant_months / (num(job.get("minimum_total_experience_years", 0)) * 12), 1.0)
                               if relevant_months is not None and num(job.get("minimum_total_experience_years", 0)) > 0 else None),
            "uncertain_duration_records": [r for r in records if r.get("duration_months") is None],
            "relevant_roles": relevant_records,
            "note": "Activity-relevant duration uses dated professional and internship roles whose title or supporting text matches the JD activity. Leadership, project, and unknown categories are excluded; relevance may still require human review.",
        }
    else:
        y = num(row.get("total_experience_years_final", row.get("total_experience_years", np.nan)), np.nan)
        row["_experience_evidence"] = {
            **common_evidence,
            "experience_requirement_type": "total_professional",
            "duration_ratio_numerator_months": total_months,
            "duration_ratio_denominator_months": minimum * 12,
            "duration_ratio": min(total_years / minimum, 1.0) if total_years is not None and minimum > 0 else None,
            "uncertain_duration_records": [r for r in records if r.get("duration_months") is None],
            "note": "Total professional duration includes dated full-time, part-time, contract/freelance, and internship roles; student leadership, projects, and unclassified roles are reported separately and excluded.",
        }
    if np.isnan(y): return None
    ratio = float(np.clip(max(0.0, y) / minimum, 0, 1))
    if isinstance(row.get("_experience_evidence"), dict):
        row["_experience_evidence"]["duration_ratio"] = ratio
    return ratio

def score_education(row, job):
    degree_raw = clean(row.get("primary_degree", row.get("degree", "")))
    course_raw = clean(row.get("course_final", row.get("course", "")))
    degree, course = norm(degree_raw), norm(course_raw)
    ad = {norm(x) for x in clean(job["accepted_degrees"]).split("|") if norm(x)}
    ac = {norm(x) for x in clean(job["accepted_courses"]).split("|") if norm(x)}
    pdg, pc = norm(job["preferred_degree"]), norm(job["preferred_course"])
    contextual = job.get("education_requirements")
    if isinstance(contextual, list) and contextual:
        requirement = contextual[0]
        level = clean(row.get("degree_level", "unknown"))
        subject = norm(row.get("degree_subject", ""))
        required_subjects = {norm(value) for value in requirement.get("subjects", [])}
        level_required = requirement.get("level")
        preference = requirement.get("preference", "uncertain")
        subject_match = bool(subject and any(subject == item or subject in item or item in subject for item in required_subjects))
        subject_mismatch = bool(subject and required_subjects and not subject_match)
        if not degree:
            value, status, reason = None, "missing_candidate_education", "No degree evidence was extracted from the resume."
        elif required_subjects and not subject:
            value, status, reason = None, "subject_unavailable", "Degree title was extracted, but the field of study could not be established."
        elif subject_mismatch:
            value, status, reason = 0.0, "explicit_subject_mismatch", "The extracted degree subject does not match any subject named by the JD."
        elif preference == "mandatory" and level in {"master", "integrated_master", "doctorate"} and level_required == "bachelor":
            bachelor_evidence = next((item for item in row.get("degree_credentials", [])
                                      if item.get("degree_level") == "bachelor"
                                      and item.get("degree_subject")
                                      and any(norm(item["degree_subject"]) == wanted or norm(item["degree_subject"]) in wanted or wanted in norm(item["degree_subject"]) for wanted in required_subjects)), None)
            if bachelor_evidence:
                value, status, reason = 1.0, "mandatory_bachelors_explicitly_evidenced", "A separate bachelor's credential with a matching subject is explicitly present in the resume."
                row["_matched_education_credential"] = bachelor_evidence
            else:
                value, status, reason = None, "mandatory_level_unverified", "A higher-level degree is evidenced, but it does not establish that the mandatory bachelor's credential was completed."
        elif preference == "preferred" and level in {"master", "integrated_master", "doctorate"} and subject_match:
            value, status, reason = 0.8, "preferred_subject_match_higher_level", "The JD prefers a bachelor's degree; a higher-level credential with a matching subject receives partial credit without treating levels as equivalent."
        elif level == level_required and (subject_match or not required_subjects):
            value, status, reason = 1.0, "required_level_and_subject_match", "The extracted degree level and subject match the JD's stated education criterion."
        elif preference == "preferred" and level in {"master", "integrated_master", "doctorate"} and not required_subjects:
            value, status, reason = 0.8, "preferred_higher_level_subject_unspecified", "A higher-level credential is evidenced for a preferred bachelor's-level criterion; the JD specified no subject."
        else:
            value, status, reason = 0.0, "explicit_level_mismatch", "The degree subject matches, but the extracted degree level does not match the JD criterion."
        row["_education_evidence"] = {
            **{key: clean(row.get(key, "")) for key in ("degree_title", "degree_level", "degree_subject", "degree_specialization", "degree_evidence")},
            "degree_credentials": row.get("degree_credentials", []),
            "matched_credential": row.get("_matched_education_credential", {}),
            "jd_preference": preference, "required_level": level_required,
            "required_subjects": requirement.get("subjects", []), "jd_source_text": requirement.get("source_text", ""),
            "score": value, "status": status, "explanation": reason,
        }
        if not pc and not ac:
            return value
    def one(v, pref, accepted):
        if not pref and not accepted: return None
        if not v: return None
        if v == pref: return 1.0
        if v in accepted: return 0.8
        if any(a and (a in v or v in a) for a in accepted): return 0.8
        return 0.0

    contextual_degree_score = value if isinstance(contextual, list) and contextual else one(degree, pdg, ad)
    components = [x for x in (contextual_degree_score, one(course,pc,ac)) if x is not None]
    return sum(components) / len(components) if components else None

def project_text(row):
    explicit_section = clean(row.get("project_evidence_text", ""))
    if explicit_section:
        return explicit_section
    vals = parse_list(row.get("projects_extracted",""))
    if vals: return " ".join(vals)
    # Fallback: extract explicit project-title blocks from resume text.
    text = clean(row.get("resume_text",""))
    found=[]
    for m in re.finditer(
        r"(?:project title|project)\s*[:\-]\s*(.*?)(?="
        r"\brole\s*:|\btools and technologies\s*:|\bproject title\s*:|"
        r"\beducation\b|\bskills?\b|$)", text, re.I|re.S):
        found.append(m.group(1).strip())
    return " ".join(found)

def jd_text(job):
    # Upload evaluations can supply the user's full JD without restructuring
    # its semantic text. Existing CSV scoring has no override key, so its
    # established behavior remains unchanged.
    if clean(job.get("semantic_text_override", "")):
        return clean(job["semantic_text_override"])
    return (
        f"Job title: {clean(job['job_title'])}. "
        f"Required skills: {clean(job['required_skills'])}. "
        f"Preferred degree: {clean(job['preferred_degree'])}. "
        f"Accepted degrees: {clean(job['accepted_degrees'])}. "
        f"Preferred course: {clean(job['preferred_course'])}. "
        f"Accepted courses: {clean(job['accepted_courses'])}. "
        f"Minimum experience: {clean(job['minimum_total_experience_years'])} years."
    )

def cosine01(a,b,model):
    if not clean(a) or not clean(b): return 0.0
    e = model.encode([clean(a),clean(b)], normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
    c = float(cosine_similarity([e[0]],[e[1]])[0,0])
    return float(np.clip((c+1)/2,0,1))

def patch_text(base_row, cfrow):
    text = clean(base_row.get("resume_text",""))
    old = clean(cfrow.get("original_value",""))
    new = clean(cfrow.get("counterfactual_value",""))
    attrs = [a.strip() for a in clean(cfrow.get("changed_attribute","")).split("+") if a.strip()]
    oldv = [x.strip() for x in old.split("|")] if len(attrs)>1 else [old]
    newv = [x.strip() for x in new.split("|")] if len(attrs)>1 else [new]
    for i, attr in enumerate(attrs):
        o = oldv[i] if i < len(oldv) else old
        v = newv[i] if i < len(newv) else new
        if o and v:
            text = re.sub(re.escape(o), v, text, flags=re.I)
    # Explicitly expose the changed proxy even if it wasn't present in text.
    meta=[]
    for i, attr in enumerate(attrs):
        v = newv[i] if i < len(newv) else new
        meta.append(f"{attr}: {v}")
    return text + ("\nCounterfactual attributes: " + "; ".join(meta) if meta else "")

def apply_cf(base_row, cfrow):
    row = base_row.copy()
    attrs = [a.strip() for a in clean(cfrow["changed_attribute"]).split("+") if a.strip()]
    vals = [x.strip() for x in clean(cfrow["counterfactual_value"]).split("|")] if len(attrs)>1 else [clean(cfrow["counterfactual_value"])]
    for i, attr in enumerate(attrs):
        v = vals[i] if i < len(vals) else vals[0]
        if attr == "name": row["name"] = v
        elif attr == "college": row["college"] = v
        elif attr == "city": row["city"] = v
        elif attr == "age": row["age_final"] = num(v, row.get("age_final",0))
        elif attr == "experience":
            months = num(v, row.get("total_experience_months_final",0))
            row["total_experience_months_final"] = months
            row["total_experience_years_final"] = months/12.0
        elif attr == "degree": row["primary_degree"] = v
        elif attr == "course": row["course_final"] = v
    row["_patched_text"] = patch_text(base_row, cfrow)
    return row

def score(row, job, model, *, skill_category_weights=None):
    # Materialize requirement classifications once so batch and upload scoring
    # use the same conservative JD interpretation.
    if not job.get("requirements"):
        raw_jd = clean(job.get("job_description", "")) or clean(job.get("semantic_text_override", ""))
        job["requirements"] = classify_jd_requirements(raw_jd)
    sem = cosine01(row.get("_patched_text",row.get("resume_text","")), jd_text(job), model)
    proj_text = project_text(row)
    proj = cosine01(proj_text, jd_text(job), model) if proj_text else 0.0
    sk = score_skills(row, job, category_weights=skill_category_weights)
    ex = score_experience(row,job)
    ed = score_education(row,job)

    # v3: requirements absent from the JD and criterion scores unavailable due
    # to missing evidence are excluded. A criterion present but not demonstrated
    # remains a real zero. Existing component weights are retained unchanged.
    values = {"skills": sk, "experience": ex, "education_course": ed,
              "projects": proj if proj_text else None, "semantic": sem}
    available = {k: float(v) for k, v in values.items() if v is not None}
    denom = sum(WEIGHTS[k] for k in available)
    effective = {k: (WEIGHTS[k] / denom if k in available and denom else 0.0) for k in WEIGHTS}
    contributions = {k: round(effective[k] * available[k], 6) if k in available else 0.0 for k in WEIGHTS}
    final = sum(contributions.values())

    return {
        "score_methodology_version": UPLOAD_METHOD_VERSION if skill_category_weights is not None else METHOD_VERSION,
        "skills_score": round(sk,6) if sk is not None else None,
        "experience_score": round(ex,6) if ex is not None else None,
        "education_course_score": round(ed,6) if ed is not None else None,
        "projects_score": round(proj,6) if proj_text else None,
        "semantic_score": round(sem,6),
        "final_score": round(float(np.clip(final,0,1)),6)
        ,"effective_weights": {k: round(v, 6) for k, v in effective.items()},
        "configured_weights": dict(WEIGHTS),
        "weighted_contributions": contributions,
        "score_denominator_weight": round(denom, 6),
        "score_formula": ("v4 upload: within skills, required criteria weight 1.0 and preferred criteria 0.5; other factor weights and available-factor normalization are unchanged." if skill_category_weights is not None else "v3: sum(configured_weight * available_component) / sum(configured_weight for available components); absent requirements and insufficient criterion evidence are excluded; configured weights unchanged."),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default="final/final_resumes.csv")
    ap.add_argument("--jobs",default="job_requirements_complete.csv")
    ap.add_argument("--counterfactual",default="final/counterfactual_resumes_valid.csv")
    ap.add_argument("--outdir",default="screening_output_final")
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)

    base=pd.read_csv(args.base)
    jobs=pd.read_csv(args.jobs)
    cf=pd.read_csv(args.counterfactual)

    jobmap=jobs.set_index("job_id").to_dict("index")
    base["job_id"]=base["source_category"].map(
        dict(zip(jobs["source_category"],jobs["job_id"]))
    )

    # All 25 should now map.
    unmapped=base[base["job_id"].isna()][["resume_id","source_category"]].drop_duplicates()
    unmapped.to_csv(os.path.join(args.outdir,"unmapped_resume_categories.csv"),index=False)

    model=SentenceTransformer(MODEL_NAME)
    supported=base[base["job_id"].notna()].copy()
    lookup={r["resume_id"]:r for _,r in supported.iterrows()}

    base_scores={}
    base_details={}
    print(f"Scoring {len(supported)} base resumes...")
    for rid,row in lookup.items():
        r=score(row,jobmap[row["job_id"]],model)
        base_scores[rid]=r["final_score"]
        base_details[rid]=r

    cf2=cf.merge(
        supported[["resume_id","job_id"]],
        left_on="original_resume_id",
        right_on="resume_id",
        how="inner",
        suffixes=("","_base")
    )

    main_rows=[]
    detailed=[]
    print(f"Scoring {len(cf2)} counterfactual variants...")
    for _,c in cf2.iterrows():
        rid=c["original_resume_id"]
        row=apply_cf(lookup[rid],c)
        r=score(row,jobmap[c["job_id"]],model)
        b=base_details[rid]
        bs=base_scores[rid]
        main_rows.append({
            "resume_id":rid,
            "job_id":c["job_id"],
            "variant_id":clean(c["cf_id"]),
            "proxy_type":clean(c["changed_attribute"]),
            "proxy_value":clean(c["counterfactual_value"]),
            "base_score":bs,
            "variant_score":r["final_score"],
        })
        detailed.append({
            "resume_id":rid,"job_id":c["job_id"],
            "variant_id":clean(c["cf_id"]),
            "proxy_type":clean(c["changed_attribute"]),
            "proxy_value":clean(c["counterfactual_value"]),
            "base_skills_score":b["skills_score"],
            "base_experience_score":b["experience_score"],
            "base_education_course_score":b["education_course_score"],
            "base_projects_score":b["projects_score"],
            "base_semantic_score":b["semantic_score"],
            "base_score":bs,
            "variant_skills_score":r["skills_score"],
            "variant_experience_score":r["experience_score"],
            "variant_education_course_score":r["education_course_score"],
            "variant_projects_score":r["projects_score"],
            "variant_semantic_score":r["semantic_score"],
            "variant_score":r["final_score"],
            "score_delta":round(r["final_score"]-bs,6),
        })

    out=pd.DataFrame(main_rows)
    det=pd.DataFrame(detailed)
    out.to_csv(os.path.join(args.outdir,"screening_results.csv"),index=False)
    det.to_csv(os.path.join(args.outdir,"screening_results_detailed.csv"),index=False)

    mapping=jobs[["job_id","source_category","job_title"]].copy()
    mapping.to_csv(os.path.join(args.outdir,"role_to_job_mapping.csv"),index=False)

    validation=pd.DataFrame([{
        "base_resumes_total":len(base),
        "base_resumes_scored":len(supported),
        "base_resumes_unmapped":len(unmapped),
        "counterfactual_total":len(cf),
        "counterfactuals_scored":len(out),
        "duplicate_variant_ids":int(out["variant_id"].duplicated().sum()) if len(out) else 0,
        "max_base_scores_per_resume":int(out.groupby("resume_id")["base_score"].nunique().max()) if len(out) else 0,
        "max_job_ids_per_resume":int(out.groupby("resume_id")["job_id"].nunique().max()) if len(out) else 0,
        "score_delta_min":float(det["score_delta"].min()) if len(det) else 0,
        "score_delta_max":float(det["score_delta"].max()) if len(det) else 0,
    }])
    validation.to_csv(os.path.join(args.outdir,"validation_report.csv"),index=False)

    # Proxy summary for quick bias-audit inspection.
    if len(det):
        summary=det.groupby("proxy_type").agg(
            variants=("variant_id","count"),
            mean_delta=("score_delta","mean"),
            median_delta=("score_delta","median"),
            min_delta=("score_delta","min"),
            max_delta=("score_delta","max"),
            mean_abs_delta=("score_delta",lambda x: np.mean(np.abs(x)))
        ).reset_index()
    else:
        summary=pd.DataFrame()
    summary.to_csv(os.path.join(args.outdir,"proxy_delta_summary.csv"),index=False)

    print("\nCOMPLETE")
    print(validation.to_string(index=False))
    print(f"\nOutput: {args.outdir}")

if __name__=="__main__":
    main()
