#!/usr/bin/env python3
"""
Explainability layer for the existing resume screening pipeline.

This script DOES NOT change the screening/scoring model. It reuses the same
screening helpers and the already-generated screening component scores to
produce recruiter-friendly, structured explanations.

Outputs:
  explanations.csv
  explanations.json
  counterfactual_explanations.csv   (optional but enabled by default)
  counterfactual_explanations.json  (optional but enabled by default)

Typical usage from the repository root:
  python explainability.py \
      --base final_resumes.csv \
      --jobs job_requirements_complete.csv \
      --detailed screening_output_final/screening_results_detailed.csv \
      --outdir explainability_output
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Mirror the existing screening helper logic locally. The explanation layer
# never recalculates or changes the final score; it only reconstructs the
# evidence behind the already-generated component scores.
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
        obj = __import__("ast").literal_eval(raw)
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
    return {norm_skill(x) for x in clean(job["required_skills"]).split("|") if norm_skill(x)}

ALIASES = {
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

def experience_years(row):
    y = num(row.get("total_experience_years_final", np.nan), np.nan)
    if not np.isnan(y): return max(0.0, y)
    m = num(row.get("total_experience_months_final", np.nan), np.nan)
    if not np.isnan(m): return max(0.0, m / 12.0)
    return max(0.0, num(row.get("total_experience_years", 0)))

def project_text(row):
    vals = parse_list(row.get("projects_extracted", ""))
    if vals: return " ".join(vals)
    text = clean(row.get("resume_text", ""))
    found = []
    for m in re.finditer(
        r"(?:project title|project)\s*[:\-]\s*(.*?)(?="
        r"\brole\s*:|\btools and technologies\s*:|\bproject title\s*:|"
        r"\beducation\b|\bskills?\b|$)", text, re.I | re.S):
        found.append(m.group(1).strip())
    return " ".join(found)


UNKNOWN_VALUES = {"", "unknown", "none", "nan", "null", "not available"}


def is_missing_value(value) -> bool:
    return clean(value).lower() in UNKNOWN_VALUES


def split_job_values(value) -> List[str]:
    return [clean(x) for x in clean(value).split("|") if clean(x)]


def education_one(value: str, preferred: str, accepted: List[str]) -> Tuple[float, str]:
    """Mirror the education rule in run_screening.score_education()."""
    v = norm(value)
    pref = norm(preferred)
    accepted_norm = {norm(x) for x in accepted if norm(x)}

    if not v or v in UNKNOWN_VALUES:
        return 0.0, "Not provided"
    if v == pref:
        return 1.0, "Preferred"
    if v in accepted_norm or any(a and (a in v or v in a) for a in accepted_norm):
        return 0.8, "Accepted"
    return 0.0, "Not accepted"


def original_skill_display(row) -> List[str]:
    """Return display-friendly resume skills while using the scorer's parsing order."""
    for col in ["skills_clean_new", "skills_extracted", "skills_clean"]:
        vals = parse_list(row.get(col, ""))
        if vals:
            return [clean(v) for v in vals if clean(v)]
    return []


def required_skill_display(job) -> List[str]:
    return [clean(x) for x in clean(job["required_skills"]).split("|") if clean(x)]


def explain_skills(row, job) -> Dict:
    required_display = required_skill_display(job)
    required_norm = [(x, norm_skill(x)) for x in required_display if norm_skill(x)]
    resume_display = original_skill_display(row)
    resume_norm = [(x, norm_skill(x)) for x in resume_display if norm_skill(x)]

    matched = []
    missing = []
    matched_pairs = []

    for req_display, req_norm in required_norm:
        matched_have = next(
            (have_display for have_display, have_norm in resume_norm if skill_match(req_norm, have_norm)),
            None,
        )
        if matched_have is not None:
            matched.append(req_display)
            matched_pairs.append({"required_skill": req_display, "resume_skill": matched_have})
        else:
            missing.append(req_display)

    score = 1.0 if not required_norm else len(matched) / len(required_norm)

    return {
        "required_skills": required_display,
        "matched_skills": matched,
        "missing_skills": missing,
        "matched_skill_pairs": matched_pairs,
        "required_skill_count": len(required_norm),
        "matched_skill_count": len(matched),
        "score": round(float(score), 6),
    }


def explain_experience(row, job) -> Dict:
    candidate_years = float(experience_years(row))
    required_years = float(pd.to_numeric(job["minimum_total_experience_years"], errors="coerce"))
    required_years = 0.0 if np.isnan(required_years) else required_years

    if required_years <= 0:
        status = "No minimum requirement"
    elif candidate_years >= required_years:
        status = "Meets requirement"
    else:
        status = "Below requirement"

    score = 1.0 if required_years <= 0 else float(np.clip(candidate_years / required_years, 0, 1))

    return {
        "candidate_years": round(candidate_years, 2),
        "required_years": round(required_years, 2),
        "difference_years": round(candidate_years - required_years, 2),
        "status": status,
        "score": round(score, 6),
    }


def explain_education_course(row, job) -> Dict:
    degree = clean(row.get("primary_degree", ""))
    course = clean(row.get("course_final", ""))

    accepted_degrees = split_job_values(job["accepted_degrees"])
    accepted_courses = split_job_values(job["accepted_courses"])

    degree_score, degree_status = education_one(degree, clean(job["preferred_degree"]), accepted_degrees)
    course_score, course_status = education_one(course, clean(job["preferred_course"]), accepted_courses)

    overall = (degree_score + course_score) / 2.0

    return {
        "degree": {
            "candidate": degree if not is_missing_value(degree) else "Not provided",
            "preferred": clean(job["preferred_degree"]),
            "accepted": accepted_degrees,
            "status": degree_status,
            "score": round(degree_score, 6),
        },
        "course": {
            "candidate": course if not is_missing_value(course) else "Not provided",
            "preferred": clean(job["preferred_course"]),
            "accepted": accepted_courses,
            "status": course_status,
            "score": round(course_score, 6),
        },
        "score": round(overall, 6),
    }


def semantic_interpretation(score: float) -> str:
    # This is only a human-readable interpretation of the already-calculated score.
    # It does not affect screening.
    if score >= 0.75:
        return "High semantic alignment"
    if score >= 0.60:
        return "Moderate-to-high semantic alignment"
    if score >= 0.45:
        return "Moderate semantic alignment"
    return "Low semantic alignment"


def project_interpretation(score: float, available: bool) -> str:
    if not available:
        return "Project evidence was unavailable, so this component was excluded from the final-score denominator."
    if score >= 0.75:
        return "High project-to-role alignment"
    if score >= 0.60:
        return "Moderate-to-high project-to-role alignment"
    if score >= 0.45:
        return "Moderate project-to-role alignment"
    return "Low project-to-role alignment"


def effective_weights(row, component_scores: Dict[str, float]) -> Dict[str, float]:
    """Mirror run_screening.score(): project weight is removed when no project text exists."""
    has_project = bool(project_text(row))
    available = {k: v for k, v in component_scores.items() if not (k == "projects" and not has_project)}
    denom = sum(WEIGHTS[k] for k in available)
    if denom <= 0:
        return {k: 0.0 for k in WEIGHTS}
    return {k: float(WEIGHTS[k] / denom) if k in available else 0.0 for k in WEIGHTS}


def component_contributions(component_scores: Dict[str, float], weights: Dict[str, float]) -> Dict[str, float]:
    return {k: round(float(component_scores[k] * weights[k]), 6) for k in component_scores}


def make_overall_explanation(candidate_id, job_title, skills, experience, education, semantic, projects, final_score, contributions):
    skill_part = (
        f"matches {skills['matched_skill_count']} of {skills['required_skill_count']} required skills"
        if skills["required_skill_count"]
        else "has no explicit required-skill list"
    )
    exp_part = (
        f"has {experience['candidate_years']:.2f} years of experience versus "
        f"{experience['required_years']:.2f} years required ({experience['status'].lower()})"
    )
    degree_status = education["degree"]["status"].lower()
    course_status = education["course"]["status"].lower()

    strongest = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    strongest_names = {
        "skills": "skills",
        "experience": "experience",
        "education_course": "education/course",
        "projects": "projects",
        "semantic": "semantic similarity",
    }
    top = [strongest_names[k] for k, v in strongest if v > 0][:2]
    top_text = ", ".join(top) if top else "the available screening components"

    return (
        f"Candidate {candidate_id} received {final_score:.4f} for {job_title}. "
        f"The resume {skill_part}; it {exp_part}. "
        f"The degree is {degree_status} and the course is {course_status}. "
        f"Semantic similarity is {semantic['score']:.4f} ({semantic['interpretation']}); "
        f"project similarity is {projects['score']:.4f}. "
        f"The largest weighted contributions come from {top_text}."
    )


def build_base_explanation(row, job, detail_row) -> Dict:
    candidate_id = clean(row["resume_id"])
    job_id = clean(row["job_id"])
    job_title = clean(job["job_title"])

    skills = explain_skills(row, job)
    experience = explain_experience(row, job)
    education = explain_education_course(row, job)

    semantic_score = float(detail_row["base_semantic_score"])
    project_score = float(detail_row["base_projects_score"])
    final_score = float(detail_row["base_score"])

    projects_available = bool(project_text(row))
    semantic = {
        "score": round(semantic_score, 6),
        "interpretation": semantic_interpretation(semantic_score),
    }
    projects = {
        "score": round(project_score, 6),
        "available": projects_available,
        "interpretation": project_interpretation(project_score, projects_available),
    }

    component_scores = {
        "skills": skills["score"],
        "experience": experience["score"],
        "education_course": education["score"],
        "projects": project_score,
        "semantic": semantic_score,
    }
    weights = effective_weights(row, component_scores)
    contributions = component_contributions(component_scores, weights)

    explanation = make_overall_explanation(
        candidate_id,
        job_title,
        skills,
        experience,
        education,
        semantic,
        projects,
        final_score,
        contributions,
    )

    return {
        "resume_id": candidate_id,
        "job_id": job_id,
        "job_title": job_title,
        "source_category": clean(row.get("source_category", "")),
        "name": clean(row.get("name", "")),
        "matched_skills": skills["matched_skills"],
        "missing_skills": skills["missing_skills"],
        "matched_skill_pairs": skills["matched_skill_pairs"],
        "required_skill_count": skills["required_skill_count"],
        "matched_skill_count": skills["matched_skill_count"],
        "skills_score": skills["score"],
        "candidate_experience_years": experience["candidate_years"],
        "required_experience_years": experience["required_years"],
        "experience_difference_years": experience["difference_years"],
        "experience_status": experience["status"],
        "experience_score": experience["score"],
        "degree": education["degree"]["candidate"],
        "degree_status": education["degree"]["status"],
        "degree_score": education["degree"]["score"],
        "course": education["course"]["candidate"],
        "course_status": education["course"]["status"],
        "course_score": education["course"]["score"],
        "education_course_score": education["score"],
        "projects_score": projects["score"],
        "projects_available": projects["available"],
        "projects_interpretation": projects["interpretation"],
        "semantic_score": semantic["score"],
        "semantic_interpretation": semantic["interpretation"],
        "effective_skill_weight": round(weights["skills"], 6),
        "effective_experience_weight": round(weights["experience"], 6),
        "effective_education_course_weight": round(weights["education_course"], 6),
        "effective_project_weight": round(weights["projects"], 6),
        "effective_semantic_weight": round(weights["semantic"], 6),
        "skills_contribution": contributions["skills"],
        "experience_contribution": contributions["experience"],
        "education_course_contribution": contributions["education_course"],
        "projects_contribution": contributions["projects"],
        "semantic_contribution": contributions["semantic"],
        "overall_score": round(final_score, 6),
        "overall_explanation": explanation,
    }


def changed_component_explanations(detail_row) -> List[Dict]:
    pairs = [
        ("skills", "base_skills_score", "variant_skills_score"),
        ("experience", "base_experience_score", "variant_experience_score"),
        ("education_course", "base_education_course_score", "variant_education_course_score"),
        ("projects", "base_projects_score", "variant_projects_score"),
        ("semantic", "base_semantic_score", "variant_semantic_score"),
    ]
    # The counterfactual variants use the same job and the same project
    # availability as their base resume, so the effective weights are the same.
    has_project = float(detail_row["base_projects_score"]) != 0.0 or float(detail_row["variant_projects_score"]) != 0.0
    available = [name for name, _, _ in pairs if name != "projects" or has_project]
    denom = sum(WEIGHTS[k] for k in available)
    eff = {k: (WEIGHTS[k] / denom if k in available else 0.0) for k in WEIGHTS} if denom else {k: 0.0 for k in WEIGHTS}

    changes = []
    for name, base_col, variant_col in pairs:
        b = float(detail_row[base_col])
        v = float(detail_row[variant_col])
        d = v - b
        weighted_d = d * eff[name]
        changes.append({
            "component": name,
            "base_score": round(b, 6),
            "variant_score": round(v, 6),
            "change": round(d, 6),
            "effective_weight": round(eff[name], 6),
            "weighted_change": round(weighted_d, 6),
        })
    return sorted(changes, key=lambda x: abs(x["weighted_change"]), reverse=True)


def build_counterfactual_explanation(cf_row) -> Dict:
    changes = changed_component_explanations(cf_row)
    meaningful = [x for x in changes if abs(x["change"]) >= 1e-6]

    if meaningful:
        primary = meaningful[0]
        if primary["weighted_change"] < 0:
            driver = f"{primary['component']} made the largest negative weighted contribution change ({primary['weighted_change']:+.4f})"
        else:
            driver = f"{primary['component']} made the largest positive weighted contribution change ({primary['weighted_change']:+.4f})"
    else:
        driver = "No screening component changed"

    return {
        "resume_id": clean(cf_row["resume_id"]),
        "job_id": clean(cf_row["job_id"]),
        "variant_id": clean(cf_row["variant_id"]),
        "proxy_type": clean(cf_row["proxy_type"]),
        "proxy_value": clean(cf_row["proxy_value"]),
        "base_score": round(float(cf_row["base_score"]), 6),
        "variant_score": round(float(cf_row["variant_score"]), 6),
        "score_delta": round(float(cf_row["score_delta"]), 6),
        "component_changes": changes,
        "primary_changed_component": changes[0]["component"] if changes else None,
        "primary_component_change": changes[0]["change"] if changes else 0.0,
        "primary_weighted_change": changes[0]["weighted_change"] if changes else 0.0,
        "explanation": (
            f"Changing {clean(cf_row['proxy_type'])} to {clean(cf_row['proxy_value'])} "
            f"changed the score from {float(cf_row['base_score']):.4f} to "
            f"{float(cf_row['variant_score']):.4f} "
            f"(delta {float(cf_row['score_delta']):+.4f}). {driver}."
        ),
    }


def flatten_for_csv(record: Dict) -> Dict:
    out = {}
    for key, value in record.items():
        if isinstance(value, (list, dict)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def main():

    parser = argparse.ArgumentParser(
        description="Generate recruiter-friendly explanations from the existing screening outputs."
    )

    parser.add_argument(
        "--base",
        default="final/final_resumes.csv"
    )

    parser.add_argument(
        "--jobs",
        default="job_requirements_complete.csv"
    )

    parser.add_argument(
        "--detailed",
        default="screening_output_final/screening_results_detailed.csv"
    )

    parser.add_argument(
        "--outdir",
        default="outputs/explainability"
    )

    parser.add_argument(
        "--skip-counterfactuals",
        action="store_true"
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base = pd.read_csv(args.base)
    jobs = pd.read_csv(args.jobs)
    detailed = pd.read_csv(args.detailed)

    if "job_id" not in base.columns:
        role_to_job = dict(zip(jobs["source_category"], jobs["job_id"]))
        base["job_id"] = base["source_category"].map(role_to_job)

    job_lookup = jobs.set_index("job_id").to_dict("index")
    base_lookup = base.set_index("resume_id").to_dict("index")

    # Exactly one base explanation per scored candidate.
    base_detail = detailed.drop_duplicates("resume_id").set_index("resume_id")
    records = []
    missing_details = []

    for rid, row in base_lookup.items():
        row["resume_id"] = rid
        if rid not in base_detail.index:
            missing_details.append(rid)
            continue
        job_id = clean(row.get("job_id", ""))
        if not job_id or job_id not in job_lookup:
            continue
        records.append(build_base_explanation(row, job_lookup[job_id], base_detail.loc[rid]))

    # Stable ordering for UI/database consumption.
    records = sorted(records, key=lambda x: x["resume_id"])

    with open(os.path.join(args.outdir, "explanations.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    pd.DataFrame([flatten_for_csv(x) for x in records]).to_csv(
        os.path.join(args.outdir, "explanations.csv"), index=False
    )

    summary = pd.DataFrame([{
        "base_candidates": len(base),
        "explanations_generated": len(records),
        "missing_screening_details": len(missing_details),
        "explanation_coverage": round(len(records) / len(base), 6) if len(base) else 0.0,
    }])
    summary.to_csv(os.path.join(args.outdir, "explainability_validation.csv"), index=False)

    if missing_details:
        pd.DataFrame({"resume_id": missing_details}).to_csv(
            os.path.join(args.outdir, "missing_explanation_candidates.csv"), index=False
        )

    if not args.skip_counterfactuals:
        cf_records = [build_counterfactual_explanation(r) for _, r in detailed.iterrows()]
        with open(os.path.join(args.outdir, "counterfactual_explanations.json"), "w", encoding="utf-8") as f:
            json.dump(cf_records, f, indent=2, ensure_ascii=False)
        pd.DataFrame([flatten_for_csv(x) for x in cf_records]).to_csv(
            os.path.join(args.outdir, "counterfactual_explanations.csv"), index=False
        )

    print("Explainability generation complete.")
    print(summary.to_string(index=False))
    print(f"Output directory: {args.outdir}")


if __name__ == "__main__":
    main()
