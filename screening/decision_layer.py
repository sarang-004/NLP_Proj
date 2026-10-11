"""Screening & Decision Layer for NLP_Proj.

This layer reuses run_screening.score() as the canonical scoring function.
It is independent of audit/ and exports both a rich decision output and the
legacy audit-compatible screening_results_detailed.csv schema.

Run from the repository root:
  python data_layer.py
  python screening/decision_layer.py --job-id JOB007
  python screening/decision_layer.py --job-id JOB007 --resume-id R001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_screening import (  # noqa: E402
    MODEL_NAME,
    WEIGHTS,
    SentenceTransformer,
    clean,
    norm,
    num,
    parse_list,
    required_skills,
    resume_skills,
    skill_match,
    experience_years,
    project_text,
    score,
    patch_text,
)

MISSING = {"", "NA", "N/A", "NAN", "NONE", "NULL"}


def present(value: Any) -> bool:
    return clean(value).strip().upper() not in MISSING


def first_value(row: pd.Series, columns: list[str], default: Any = "") -> Any:
    for col in columns:
        if col in row.index and present(row.get(col)):
            return row.get(col)
    return default


def load_resumes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Resume input not found: {path}. Run `python data_layer.py` or pass --base.")
    df = pd.read_csv(path, keep_default_na=False)
    if "resume_id" not in df.columns:
        raise ValueError("Resume input must contain resume_id.")
    if df["resume_id"].astype(str).duplicated().any():
        raise ValueError("resume_id must be unique in the base resume input.")

    aliases = {
        "primary_degree": ["primary_degree", "degree"],
        "course_final": ["course_final", "course", "specialization"],
        "skills_clean_new": ["skills_clean_new", "skills_extracted", "skills_clean", "skills"],
        "projects_extracted": ["projects_extracted", "projects"],
        "total_experience_years_final": ["total_experience_years_final", "total_experience_years"],
        "age_final": ["age_final", "age"],
    }
    for target, choices in aliases.items():
        if target not in df.columns:
            df[target] = df.apply(lambda row: first_value(row, choices, ""), axis=1)
    for col in ["resume_text", "name", "source_category", "college", "college_tier", "city", "city_tier",
                "degree", "course", "specialization", "employment_gap_months"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["primary_degree", "course_final", "skills_clean_new", "projects_extracted", "resume_text"]:
        df[col] = df[col].apply(lambda value: "" if not present(value) else value)
    return df


def load_jobs(path: Path, legacy_path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Job input not found: {path}")
    jobs = pd.read_csv(path, keep_default_na=False)
    if "job_id" not in jobs.columns:
        raise ValueError("Job input must contain job_id.")

    aliases = {
        "source_category": ["source_category"],
        "job_title": ["job_title", "title"],
        "preferred_degree": ["preferred_degree"],
        "accepted_degrees": ["accepted_degrees"],
        "preferred_course": ["preferred_course", "preferred_courses"],
        "accepted_courses": ["accepted_courses"],
        "minimum_total_experience_years": ["minimum_total_experience_years", "minimum_experience"],
        "required_skills": ["required_skills"],
    }
    for target, choices in aliases.items():
        if target not in jobs.columns:
            jobs[target] = jobs.apply(lambda row: first_value(row, choices, ""), axis=1)

    # data_layer.py's standardized JD currently omits source_category; recover the
    # project-defined mapping from the legacy JD table when available.
    if not jobs["source_category"].apply(present).any() and legacy_path.exists():
        legacy = pd.read_csv(legacy_path, keep_default_na=False)
        if {"job_id", "source_category"}.issubset(legacy.columns):
            mapping = legacy.drop_duplicates("job_id").set_index("job_id")["source_category"]
            jobs["source_category"] = jobs["job_id"].map(mapping).fillna("")

    needed = ["job_title", "preferred_degree", "accepted_degrees", "preferred_course",
              "accepted_courses", "minimum_total_experience_years", "required_skills"]
    for col in needed:
        if col not in jobs.columns:
            jobs[col] = ""
        jobs[col] = jobs[col].apply(lambda value: "" if not present(value) else value)
    jobs["minimum_total_experience_years"] = pd.to_numeric(
        jobs["minimum_total_experience_years"], errors="coerce"
    ).fillna(0.0)
    if jobs["job_id"].astype(str).duplicated().any():
        raise ValueError("job_id must be unique in the selected JD input.")
    return jobs


def qualification_score(value: Any, preferred: Any, accepted: Any) -> float:
    """Mirror run_screening.score_education's one-field preference rule."""
    value_n, preferred_n = norm(value), norm(preferred)
    accepted_set = {norm(item) for item in clean(accepted).split("|") if present(item)}
    if not value_n:
        return 0.0
    if preferred_n and value_n == preferred_n:
        return 1.0
    if value_n in accepted_set:
        return 0.8
    if any(item and (item in value_n or value_n in item) for item in accepted_set):
        return 0.8
    return 0.0


def synchronize_variant_fields(variant: pd.Series, changed_attribute: str) -> pd.Series:
    """Keep data-layer canonical fields and run_screening compatibility aliases aligned."""
    row = variant.copy()
    attrs = [part.strip() for part in clean(changed_attribute).split("+") if part.strip()]
    for attr in attrs:
        if attr == "degree":
            row["primary_degree"] = row.get("degree", "")
        elif attr in {"course", "specialization"}:
            value = row.get(attr, row.get("course", row.get("specialization", "")))
            row["course_final"] = value
            row["course"] = value
            row["specialization"] = value
        elif attr in {"experience", "total_experience_years"}:
            value = row.get("total_experience_years", row.get("total_experience_years_final", ""))
            row["total_experience_years_final"] = value
        elif attr == "age":
            row["age_final"] = row.get("age", "")
    return row


def factor_evidence(row: pd.Series, job: pd.Series, scored: dict[str, Any]) -> dict[str, Any]:
    have = resume_skills(row)
    required = sorted(required_skills(job))
    matched_pairs, missing = [], []
    for requirement in required:
        matches = sorted(skill for skill in have if skill_match(requirement, skill))
        if matches:
            matched_pairs.append({"required_skill": requirement, "resume_skill": matches[0]})
        else:
            missing.append(requirement)

    degree = first_value(row, ["primary_degree", "degree"], "")
    course = first_value(row, ["course_final", "course", "specialization"], "")
    degree_score = qualification_score(degree, job.get("preferred_degree", ""), job.get("accepted_degrees", ""))
    course_score = qualification_score(course, job.get("preferred_course", ""), job.get("accepted_courses", ""))
    candidate_exp = experience_years(row)
    required_exp = num(job.get("minimum_total_experience_years", 0))
    projects = project_text(row)
    project_available = bool(clean(projects))

    return {
        "matched_skills": [pair["required_skill"] for pair in matched_pairs],
        "missing_skills": missing,
        "matched_skill_pairs": matched_pairs,
        "required_skill_count": len(required),
        "matched_skill_count": len(matched_pairs),
        "candidate_experience_years": round(candidate_exp, 4),
        "required_experience_years": round(required_exp, 4),
        "experience_status": "Meets requirement" if required_exp <= 0 or candidate_exp >= required_exp else "Below requirement",
        "experience_evidence": f"{candidate_exp:.2f} years reported; {required_exp:.2f} years required",
        "degree": clean(degree) if present(degree) else "Not provided",
        "degree_score": round(degree_score, 6),
        "degree_status": "Not provided" if not present(degree) else "Preferred" if degree_score == 1 else "Accepted" if degree_score == 0.8 else "Not accepted",
        "course": clean(course) if present(course) else "Not provided",
        "course_score": round(course_score, 6),
        "course_status": "Not provided" if not present(course) else "Preferred" if course_score == 1 else "Accepted" if course_score == 0.8 else "Not accepted",
        "projects_available": project_available,
        "project_evidence": clean(projects) if project_available else "No project evidence extracted",
        "semantic_evidence": "Resume-to-JD semantic similarity; this similarity signal is not a causal explanation.",
    }


def evaluate(row: pd.Series, job: pd.Series, model: Any) -> dict[str, Any]:
    scored = score(row, job, model)  # Canonical scoring; do not recalculate final_score here.
    evidence = factor_evidence(row, job, scored)

    available_weights = {
        key: weight for key, weight in WEIGHTS.items()
        if not (key == "projects" and not evidence["projects_available"])
    }
    denominator = sum(available_weights.values())
    effective = {key: value / denominator for key, value in available_weights.items()}
    effective.setdefault("projects", 0.0)

    degree_contribution = effective["education_course"] * evidence["degree_score"] / 2
    course_contribution = effective["education_course"] * evidence["course_score"] / 2
    output = {
        "resume_id": clean(row.get("resume_id")),
        "job_id": clean(job.get("job_id")),
        "job_title": clean(job.get("job_title")),
        "name": clean(row.get("name")),
        "source_category": clean(row.get("source_category")),
        "skill_score": scored["skills_score"],
        "experience_score": scored["experience_score"],
        "education_score": evidence["degree_score"],
        "course_score": evidence["course_score"],
        "education_course_score": scored["education_course_score"],
        "project_score": scored["projects_score"],
        "semantic_score": scored["semantic_score"],
        "final_score": scored["final_score"],
        "skill_contribution": round(effective["skills"] * scored["skills_score"], 6),
        "experience_contribution": round(effective["experience"] * scored["experience_score"], 6),
        "education_contribution": round(degree_contribution, 6),
        "course_contribution": round(course_contribution, 6),
        "education_course_contribution": round(effective["education_course"] * scored["education_course_score"], 6),
        "project_contribution": round(effective["projects"] * scored["projects_score"], 6),
        "semantic_contribution": round(effective["semantic"] * scored["semantic_score"], 6),
        "effective_skill_weight": round(effective["skills"], 6),
        "effective_experience_weight": round(effective["experience"], 6),
        "effective_education_course_weight": round(effective["education_course"], 6),
        "effective_project_weight": round(effective["projects"], 6),
        "effective_semantic_weight": round(effective["semantic"], 6),
        **evidence,
    }
    contribution_sum = sum(output[key] for key in [
        "skill_contribution", "experience_contribution", "education_course_contribution",
        "project_contribution", "semantic_contribution"
    ])
    output["contribution_sum"] = round(contribution_sum, 6)
    output["contribution_score_difference"] = round(output["final_score"] - contribution_sum, 6)
    output["decision_note"] = "Ranking score only; review supporting evidence before a hiring decision."
    return output


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidates and export audit-compatible screening results.")
    parser.add_argument("--base", default="data/structured_resumes.csv")
    parser.add_argument("--jobs", default="data/structured_job_descriptions.csv")
    parser.add_argument("--counterfactual", default="data/structured_counterfactuals.csv")
    parser.add_argument("--legacy-jobs", default="job_requirements_complete.csv")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--resume-id", default=None)
    parser.add_argument("--outdir", default="screening_output_final")
    parser.add_argument("--skip-counterfactuals", action="store_true")
    args = parser.parse_args()

    base_path, jobs_path = (ROOT / args.base).resolve(), (ROOT / args.jobs).resolve()
    cf_path, legacy_path = (ROOT / args.counterfactual).resolve(), (ROOT / args.legacy_jobs).resolve()
    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    resumes, jobs = load_resumes(base_path), load_jobs(jobs_path, legacy_path)
    job_by_id = {str(row["job_id"]): row for _, row in jobs.iterrows()}
    category_to_job = {
        str(row["source_category"]): str(row["job_id"])
        for _, row in jobs.iterrows() if present(row.get("source_category", ""))
    }

    if args.job_id:
        if str(args.job_id) not in job_by_id:
            raise ValueError(f"Unknown job_id {args.job_id!r}; available IDs: {sorted(job_by_id)}")
        selected_jobs = {str(args.job_id): job_by_id[str(args.job_id)]}
        base_candidates = resumes.copy()
        base_candidates["__job_id"] = str(args.job_id)
    else:
        base_candidates = resumes[resumes["source_category"].astype(str).isin(category_to_job)].copy()
        base_candidates["__job_id"] = base_candidates["source_category"].astype(str).map(category_to_job)
        selected_jobs = job_by_id
        if base_candidates.empty:
            raise ValueError("No resumes mapped to jobs. Pass --job-id or check source_category mapping.")

    if args.resume_id:
        base_candidates = base_candidates[base_candidates["resume_id"].astype(str) == str(args.resume_id)]
        if base_candidates.empty:
            raise ValueError(f"resume_id {args.resume_id!r} was not found in the selected candidate set.")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    base_result_by_id: dict[str, dict[str, Any]] = {}
    base_row_by_id: dict[str, pd.Series] = {}
    base_job_by_id: dict[str, str] = {}
    base_records = []

    for _, raw_row in base_candidates.iterrows():
        row = raw_row.copy()
        job_id = str(row["__job_id"])
        job = pd.Series(job_by_id[job_id])
        result = evaluate(row, job, model)
        result["rank_scope_job_id"] = job_id
        base_records.append(result)
        base_result_by_id[str(row["resume_id"])] = result
        base_row_by_id[str(row["resume_id"])] = row
        base_job_by_id[str(row["resume_id"])] = job_id

    base_results = pd.DataFrame(base_records)
    base_results = base_results.sort_values(["job_id", "final_score", "resume_id"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    base_results["rank"] = base_results.groupby("job_id")["final_score"].rank(method="first", ascending=False).astype(int)

    cf_records: list[dict[str, Any]] = []
    cf_validation: list[dict[str, Any]] = []
    if not args.skip_counterfactuals and cf_path.exists():
        counterfactuals = pd.read_csv(cf_path, keep_default_na=False)
        required_cf_cols = {"original_resume_id", "changed_attribute", "original_value", "counterfactual_value"}
        if not required_cf_cols.issubset(counterfactuals.columns):
            raise ValueError(f"Counterfactual input is missing columns: {sorted(required_cf_cols - set(counterfactuals.columns))}")
        for _, cf in counterfactuals.iterrows():
            parent_id = str(cf["original_resume_id"])
            if parent_id not in base_row_by_id:
                continue
            job_id = base_job_by_id[parent_id]
            if args.job_id and job_id != str(args.job_id):
                job_id = str(args.job_id)
            parent = base_row_by_id[parent_id]
            variant = cf.copy()
            # structured_counterfactuals.csv carries the complete variant row.
            # If a compact counterfactual is supplied, fall back to changing the declared field.
            if "resume_text" not in variant or not present(variant.get("resume_text", "")):
                variant = parent.copy()
                attr = clean(cf["changed_attribute"])
                new_value = cf["counterfactual_value"]
                field_map = {
                    "experience": "total_experience_years", "degree": "degree",
                    "course": "course", "specialization": "specialization",
                    "age": "age", "name": "name", "college": "college",
                    "college_tier": "college_tier", "city": "city", "city_tier": "city_tier",
                    "employment_gap_months": "employment_gap_months",
                }
                for attr_part, new_part in zip(attr.split("+"), clean(new_value).split("|")):
                    field = field_map.get(attr_part.strip(), attr_part.strip())
                    variant[field] = new_part
            variant = synchronize_variant_fields(pd.Series(variant), cf["changed_attribute"])
            # Do not let missing-value sentinels such as "NA" be treated as literal
            # text to replace inside the resume (which could corrupt unrelated words).
            old_parts = str(cf["original_value"]).split("|")
            safe_old = "|".join(part if present(part) else "" for part in old_parts)
            patch_meta = pd.Series({
                "changed_attribute": cf["changed_attribute"],
                "original_value": safe_old,
                "counterfactual_value": cf["counterfactual_value"],
            })
            variant["_patched_text"] = patch_text(parent, patch_meta)
            job = pd.Series(job_by_id[job_id])
            variant_score = evaluate(variant, job, model)
            base_score = base_result_by_id[parent_id]
            variant_id = clean(cf.get("counterfactual_id", cf.get("cf_id", ""))) or f"{parent_id}_CF_{len(cf_records)+1}"
            record = {
                "resume_id": parent_id,
                "job_id": job_id,
                "variant_id": variant_id,
                "proxy_type": clean(cf["changed_attribute"]),
                "proxy_value": clean(cf["counterfactual_value"]),
                "original_value": clean(cf["original_value"]),
                "counterfactual_value": clean(cf["counterfactual_value"]),
                "base_skills_score": base_score["skill_score"],
                "base_experience_score": base_score["experience_score"],
                "base_education_course_score": base_score["education_course_score"],
                "base_projects_score": base_score["project_score"],
                "base_semantic_score": base_score["semantic_score"],
                "base_score": base_score["final_score"],
                "variant_skills_score": variant_score["skill_score"],
                "variant_experience_score": variant_score["experience_score"],
                "variant_education_course_score": variant_score["education_course_score"],
                "variant_projects_score": variant_score["project_score"],
                "variant_semantic_score": variant_score["semantic_score"],
                "variant_score": variant_score["final_score"],
                "score_delta": round(variant_score["final_score"] - base_score["final_score"], 6),
                "skills_delta": round(variant_score["skill_score"] - base_score["skill_score"], 6),
                "experience_delta": round(variant_score["experience_score"] - base_score["experience_score"], 6),
                "education_course_delta": round(variant_score["education_course_score"] - base_score["education_course_score"], 6),
                "projects_delta": round(variant_score["project_score"] - base_score["project_score"], 6),
                "semantic_delta": round(variant_score["semantic_score"] - base_score["semantic_score"], 6),
                "base_education_score": base_score["education_score"],
                "variant_education_score": variant_score["education_score"],
                "base_course_score": base_score["course_score"],
                "variant_course_score": variant_score["course_score"],
                "changed_attribute": clean(cf["changed_attribute"]),
                "base_matched_skills": json.dumps(base_score["matched_skills"], ensure_ascii=False),
                "variant_matched_skills": json.dumps(variant_score["matched_skills"], ensure_ascii=False),
                "base_missing_skills": json.dumps(base_score["missing_skills"], ensure_ascii=False),
                "variant_missing_skills": json.dumps(variant_score["missing_skills"], ensure_ascii=False),
                "experience_evidence": variant_score["experience_evidence"],
                "degree_status": variant_score["degree_status"],
                "course_status": variant_score["course_status"],
                "project_evidence": variant_score["project_evidence"],
                "semantic_evidence": variant_score["semantic_evidence"],
            }
            cf_records.append(record)
            # Validate single-attribute variants; combined changes are allowed and marked for intersectional analysis.
            declared = [x.strip() for x in clean(cf["changed_attribute"]).split("+") if x.strip()]
            if len(declared) == 1:
                expected = declared[0]
                original_struct = parent
                variant_struct = variant
                aliases = {
                    "experience": ["total_experience_years", "total_experience_years_final"],
                    "total_experience_years": ["total_experience_years", "total_experience_years_final"],
                    "degree": ["degree", "primary_degree"],
                    "course": ["course", "specialization", "course_final"],
                    "specialization": ["course", "specialization", "course_final"],
                    "age": ["age", "age_final"],
                    "employment_gap_months": ["employment_gap_months"],
                }
                allowed_fields = set(aliases.get(expected, [expected]))

                # Exclude identifiers, join keys, and counterfactual bookkeeping.
                # These are not candidate attributes and must not trigger an
                # attribute-isolation failure when a variant is synchronized.
                metadata_fields = {
                    "resume_id", "original_resume_id", "counterfactual_id",
                    "cf_id", "variant_id", "job_id", "__job_id",
                    "changed_attribute", "original_value", "counterfactual_value",
                    "_patched_text",
                }
                candidate_fields = set(original_struct.index) - metadata_fields
                changed_fields = {
                    col for col in candidate_fields
                    if present(original_struct.get(col, "")) != present(variant_struct.get(col, ""))
                    or clean(original_struct.get(col, "")) != clean(variant_struct.get(col, ""))
                }
                metadata_fields = {
                    "resume_id",
                    "original_resume_id",
                    "counterfactual_id",
                    "cf_id",
                    "variant_id",
                    "job_id",
                    "__job_id",
                    "changed_attribute",
                    "original_value",
                    "counterfactual_value",
                    "_patched_text",
                }

                unexpected = sorted(
                    changed_fields - allowed_fields - metadata_fields
                )
                cf_validation.append({
                    "variant_id": variant_id,
                    "changed_attribute": expected,
                    "unexpected_structured_changes": "|".join(unexpected),
                    "status": "passed" if not unexpected else "review",
                })

    # Rich, single-candidate and multi-candidate decision results.
    def serialize_lists(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for column in ["matched_skills", "missing_skills", "matched_skill_pairs"]:
            if column in output:
                output[column] = output[column].apply(lambda value: json.dumps(value, ensure_ascii=False))
        return output

    serialize_lists(base_results).to_csv(outdir / "screening_decision_results.csv", index=False)
    (outdir / "screening_decision_results.json").write_text(
        json.dumps(base_results.to_dict("records"), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # Existing audit/delta.py expects these names. Keep them stable and add detailed factor deltas.
    if cf_records:
        detailed = pd.DataFrame(cf_records)
        detailed.to_csv(outdir / "screening_results_detailed.csv", index=False)
        detailed[["resume_id", "job_id", "variant_id", "proxy_type", "proxy_value", "base_score", "variant_score"]].to_csv(
            outdir / "screening_results.csv", index=False
        )
        detailed.groupby("proxy_type").agg(
            variants=("variant_id", "count"),
            mean_delta=("score_delta", "mean"),
            median_delta=("score_delta", "median"),
            min_delta=("score_delta", "min"),
            max_delta=("score_delta", "max"),
            mean_abs_delta=("score_delta", lambda values: values.abs().mean()),
        ).reset_index().to_csv(outdir / "proxy_delta_summary.csv", index=False)
    else:
        # Avoid overwriting existing audit inputs with empty files when CF scoring was skipped.
        detailed = pd.DataFrame()
    validation = pd.DataFrame([{
        "base_candidates": len(base_results),
        "candidates_scored": len(base_results),
        "missing_resume_ids": int(base_results["resume_id"].eq("").sum()),
        "missing_job_ids": int(base_results["job_id"].eq("").sum()),
        "duplicate_resume_job_pairs": int(base_results.duplicated(["resume_id", "job_id"]).sum()),
        "max_abs_contribution_difference": float(base_results["contribution_score_difference"].abs().max()) if len(base_results) else 0.0,
        "counterfactual_variants_scored": len(cf_records),
        "counterfactual_single_attribute_checks": len(cf_validation),
        "counterfactual_validation_failures": sum(x["status"] != "passed" for x in cf_validation),
        "jobs_ranked": int(base_results["job_id"].nunique()) if len(base_results) else 0,
        "status": "passed" if (
            len(base_results) > 0
            and not base_results.duplicated(["resume_id", "job_id"]).any()
            and base_results["contribution_score_difference"].abs().max() <= 2e-5
            and all(x["status"] == "passed" for x in cf_validation)
        ) else "review",
    }])
    validation.to_csv(outdir / "screening_decision_validation.csv", index=False)
    pd.DataFrame(cf_validation, columns=["variant_id", "changed_attribute", "unexpected_structured_changes", "status"]).to_csv(
        outdir / "counterfactual_scoring_validation.csv", index=False
    )

    print("\nScreening & Decision Layer complete.")
    print(f"Candidates scored: {len(base_results)}")
    print(f"Jobs ranked: {base_results['job_id'].nunique()}")
    print(f"Counterfactuals scored: {len(cf_records)}")
    print(f"Outputs: {outdir}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
