from __future__ import annotations

import csv
import math
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import pandas as pd

from audit.case_validation import KEYS, reconcile_case_rows
from audit.context_analysis import classify_contextual_case
from audit.fairness import calculate_fairness_summaries
from audit.intersectionality import summarize_intersectionality, validate_intersectionality
from run_screening import MODEL_NAME
from screening.upload_workflow import (
    MAX_RESUME_BYTES,
    ResumeUploadError,
    evaluate_uploads,
    extract_uploaded_text,
    make_upload_job,
)
from screening.requirements import UPLOAD_METHOD_VERSION

app = Flask(__name__)
# The dashboard is a single-file template application; reload edited templates
# when the local non-debug server is running during UI iteration.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 11 * (MAX_RESUME_BYTES + 1024 * 1024)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS = PROJECT_ROOT / "outputs"
SCREENING = PROJECT_ROOT / "screening_output_final"
_UPLOAD_MODEL = None


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request is too large. Upload up to 10 resumes of 12 MB each."}), 413


def read_csv(path):
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError):
        return []


def number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def is_true(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def _key(row):
    return tuple(str(row.get(field, "")).strip() for field in KEYS)


def _component_deltas(row):
    pairs = {
        "skills": ("base_skills_score", "variant_skills_score"),
        "experience": ("base_experience_score", "variant_experience_score"),
        "education_course": ("base_education_course_score", "variant_education_course_score"),
        "projects": ("base_projects_score", "variant_projects_score"),
        "semantic": ("base_semantic_score", "variant_semantic_score"),
    }
    result = {}
    for name, (before, after) in pairs.items():
        left, right = number(row.get(before)), number(row.get(after))
        if left is not None and right is not None:
            result[name] = right - left
    return result


def load_dashboard():
    """Load current outputs, reconciling supplemental rows to scored results."""
    flips = read_csv(OUTPUTS / "direct_proxy_decision_flips.csv")
    unified = read_csv(OUTPUTS / "unified_audit.csv")
    screening = read_csv(SCREENING / "screening_results_detailed.csv")
    jobs = read_csv(PROJECT_ROOT / "data" / "structured_job_descriptions.csv")
    # Derive descriptive fairness summaries from the current structured
    # resume data in memory; dashboard reads must not trust stale generated
    # tables or write/regenerate pipeline outputs.
    try:
        fairness_frame, sensitivity_frame = calculate_fairness_summaries()
        fairness = fairness_frame.where(pd.notna(fairness_frame), None).to_dict("records")
        fairness_sensitivity = sensitivity_frame.where(pd.notna(sensitivity_frame), None).to_dict("records")
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        fairness, fairness_sensitivity = [], []
    source_cf = read_csv(PROJECT_ROOT / "data" / "structured_counterfactuals.csv")

    reconciled = reconcile_case_rows(unified, flips)
    job_lookup = {str(row.get("job_id", "")): row for row in jobs}
    authoritative_lookup = {_key(row): row for row in unified}
    cases = []
    for row in reconciled["cases"]:
        item = dict(row)
        status = item.get("case_validation_status", "invalid")
        if status != "verified":
            contextual = {
                "classification": "Requires review",
                "rationale": item.get("case_validation_reason", "Case could not be reconciled."),
                "job_requirement": "Unavailable: no matching authoritative screening row.",
                "supporting_evidence": [item.get("case_validation_reason", "Unmatched source row.")],
            }
        else:
            job = job_lookup.get(str(item.get("job_id", "")), {})
            contextual = classify_contextual_case(
                item.get("proxy_type", ""),
                item.get("original_value", ""),
                item.get("variant_value", item.get("counterfactual_value", "")),
                item.get("base_score"), item.get("variant_score"), job,
                _component_deltas(item),
            )
        item["context_classification"] = contextual["classification"]
        item["context_rationale"] = contextual["rationale"]
        item["contextual_job_requirement"] = contextual["job_requirement"]
        item["supporting_evidence"] = contextual["supporting_evidence"]
        cases.append(item)

    supplemental_validation = reconciled["supplemental_validation"]
    direct_types = {"name", "age", "college", "city"}
    threshold_labels = sorted({
        column.removeprefix("decision_flip_")
        for row in flips for column in row
        if column.startswith("decision_flip_")
    }, key=lambda value: number(value, 0))
    thresholds = {label: 0 for label in threshold_labels}
    verified_direct_rows = []
    for row in cases:
        if row.get("case_validation_status") != "verified" or row.get("proxy_type") not in direct_types:
            continue
        base_score, variant_score = number(row.get("base_score")), number(row.get("variant_score"))
        if base_score is None or variant_score is None:
            continue
        any_flip = False
        for label in threshold_labels:
            threshold = number(label)
            if threshold is None:
                continue
            base_selected = base_score >= threshold
            variant_selected = variant_score >= threshold
            row[f"decision_flip_{label}"] = base_selected != variant_selected
            row[f"base_selected_{label}"] = base_selected
            row[f"variant_selected_{label}"] = variant_selected
            if base_selected != variant_selected:
                thresholds[label] += 1
                any_flip = True
        row["any_decision_flip"] = any_flip
        if any_flip:
            verified_direct_rows.append(row)

    semantic_values = {}
    for row in unified:
        proxy = str(row.get("proxy_type", "")).split("+")[0]
        if proxy not in direct_types:
            continue
        base_semantic = number(row.get("base_semantic_score"))
        variant_semantic = number(row.get("variant_semantic_score"))
        if base_semantic is not None and variant_semantic is not None:
            semantic_values.setdefault(proxy, []).append(abs(variant_semantic - base_semantic))
    semantic = {key: sum(values) / len(values) for key, values in semantic_values.items() if values}

    expected_source = [row for row in source_cf if str(row.get("original_resume_id", "")) in {str(item.get("resume_id", "")) for item in unified}]
    scored_ids = {str(row.get("variant_id", "")) for row in unified}
    unscored_source = [row for row in expected_source if str(row.get("cf_id", "")) not in scored_ids]

    counts = dict(reconciled["counts"])
    counts["review_required_cases"] = sum(row.get("context_classification") == "Requires review" for row in cases)
    counts["source_variants_for_scored_candidates"] = len(expected_source)
    counts["source_variants_without_screening_results"] = len(unscored_source)
    counts["verified_direct_proxy_tests"] = sum(
        row.get("case_validation_status") == "verified" and row.get("proxy_type") in direct_types
        for row in cases
    )
    flip_count = len(verified_direct_rows)
    explainable = sum(bool(row.get("explanation") or row.get("primary_changed_component")) for row in unified)

    intersection_detail = pd.DataFrame()
    if screening:
        try:
            intersection_detail = validate_intersectionality(pd.DataFrame(screening).fillna(""))
        except (ValueError, KeyError, TypeError):
            intersection_detail = pd.DataFrame()
    intersection_summary = summarize_intersectionality(intersection_detail) if not intersection_detail.empty else pd.DataFrame()

    return {
        "metrics": {
            "total_cases": counts["unique_cases"],
            "verified_cases": counts["verified_cases"],
            "unmatched_cases": counts["unmatched_cases"],
            "invalid_cases": counts["invalid_cases"],
            "supplemental_conflicts": counts["supplemental_invalid_records"],
            "review_required_cases": counts["review_required_cases"],
            "source_variants_for_scored_candidates": counts["source_variants_for_scored_candidates"],
            "source_variants_without_screening_results": counts["source_variants_without_screening_results"],
            "direct_proxy_tests": counts["verified_direct_proxy_tests"],
            "decision_flips": flip_count,
            "explainability": round(explainable / len(unified) * 100, 1) if unified else 0,
        },
        "semantic": semantic,
        "thresholds": thresholds,
        "flips": verified_direct_rows,
        "cases": cases,
        "intersectionality": intersection_detail.where(pd.notna(intersection_detail), None).to_dict("records") if not intersection_detail.empty else [],
        "intersectionality_summary": intersection_summary.where(pd.notna(intersection_summary), None).to_dict("records") if not intersection_summary.empty else [],
        "fairness": fairness,
        "fairness_sensitivity": fairness_sensitivity,
    }


def _load_upload_model():
    global _UPLOAD_MODEL
    if _UPLOAD_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _UPLOAD_MODEL = SentenceTransformer(MODEL_NAME)
    return _UPLOAD_MODEL


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/dashboard")
def dashboard():
    return jsonify(load_dashboard())


@app.route("/api/evaluate", methods=["POST"])
def evaluate_uploads_route():
    files = [file for file in request.files.getlist("resumes") if file and file.filename]
    if not files:
        return jsonify({"error": "Choose at least one PDF, DOCX, or TXT resume."}), 400
    if len(files) > 10:
        return jsonify({"error": "Upload at most 10 resumes per evaluation."}), 400
    form = request.form.to_dict()
    jd_source = form.get("jd_source", "text")
    if jd_source == "pdf":
        jd_file = request.files.get("job_description_pdf")
        if not jd_file or not jd_file.filename:
            return jsonify({"error": "Choose a PDF job description, or switch to pasted text."}), 400
        if Path(jd_file.filename).suffix.lower() != ".pdf":
            return jsonify({"error": "The job description upload must be a PDF."}), 400
        try:
            form["job_description"] = extract_uploaded_text(jd_file.filename, jd_file.stream)
        except ResumeUploadError as exc:
            return jsonify({"error": f"Could not read the job description PDF: {exc}"}), 400
    elif jd_source == "text":
        if not form.get("job_description", "").strip():
            return jsonify({"error": "Paste the job description text, or choose PDF upload."}), 400
    else:
        return jsonify({"error": "Choose either pasted JD text or a JD PDF."}), 400

    try:
        job = make_upload_job(form)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    extracted, errors = [], []
    for index, uploaded in enumerate(files, start=1):
        candidate_id = f"Candidate {index}"
        try:
            text = extract_uploaded_text(uploaded.filename, uploaded.stream)
            extracted.append((candidate_id, text, Path(uploaded.filename).stem))
        except ResumeUploadError as exc:
            errors.append({"candidate_id": candidate_id, "message": str(exc)})
    if not extracted:
        return jsonify({"error": "No resumes could be evaluated.", "file_errors": errors}), 400
    try:
        candidates = evaluate_uploads(extracted, job, _load_upload_model())
    except Exception as exc:
        app.logger.exception("Uploaded resume evaluation failed")
        return jsonify({"error": f"Evaluation could not be completed: {exc}", "file_errors": errors}), 503

    warnings = []
    if not job["preferred_degree"] and not job["accepted_degrees"]:
        warnings.append("No degree requirement was provided; any extracted degree is shown as resume evidence without a numerical match score.")
    if not job["preferred_course"] and not job["accepted_courses"]:
        warnings.append("No academic course or major was specified for numerical scoring. Any preferred coursework, certification, or related project evidence is reported separately without an invented score.")
    if not job["required_skills"]:
        warnings.append("No score-eligible skill qualification was detected; the skills component is unavailable and excluded from the denominator.")
    if job["experience_requirement_source"] == "not_detected" and job["minimum_total_experience_years"] == 0:
        warnings.append("No numeric experience duration was specified in the JD; extracted resume experience is shown as evidence without an experience score.")
    elif job["minimum_total_experience_years"] > 0:
        requirements = job["requirements"].get("requirements", [])
        technology_specific = any(
            item.get("kind") == "experience_duration"
            and item.get("criterion") != "total professional experience"
            and item.get("category") in {"explicit_minimum", "preferred"}
            for item in requirements
        )
        activity_specific = any(
            item.get("kind") == "experience_duration"
            and item.get("criterion") == "total professional experience"
            and item.get("activity")
            and item.get("category") in {"explicit_minimum", "preferred"}
            for item in requirements
        )
        if technology_specific:
            warnings.append("The system cannot verify how many years were spent using a specific technology or activity unless the resume explicitly ties a duration to it; total career years are not substituted for technology-specific requirements.")
        elif activity_specific:
            warnings.append("The JD specifies a type of work as well as years. Only dated professional or internship roles with matching role or resume evidence contribute; student leadership, projects, and unrelated role dates are shown separately.")
        elif any(candidate["component_scores"]["experience"] is None for candidate in candidates):
            warnings.append("A numeric total-experience minimum was detected, but a reliable total duration could not be extracted for one or more resumes; those experience scores are unavailable.")
    weight_profiles = {tuple(sorted(candidate["effective_weights"].items())) for candidate in candidates}
    comparability = {
        "directly_comparable": len(weight_profiles) <= 1,
        "effective_weights_differ": len(weight_profiles) > 1,
        "warning": ("Candidates have different effective factor weights because unavailable factors were excluded. Their normalized scores and ranking are not directly comparable." if len(weight_profiles) > 1 else "All candidates use the same effective factor weights."),
        "ranking_interpretation": ("The displayed order is the arithmetic order of each candidate's individually normalized score. Because effective weights differ, interpret this only as a provisional ranking, not a like-for-like comparison of candidate fit." if len(weight_profiles) > 1 else "The displayed order follows the same effective factor weights for all candidates."),
    }
    if len(weight_profiles) > 1:
        warnings.append(comparability["warning"])
    return jsonify({
        "job_title": job["job_title"],
        "criteria": {
            "required_skills": job["required_skills_list"],
            "required_skills_source": job["required_skills_source"],
            "minimum_experience_years": job["minimum_total_experience_years"],
            "experience_source": job["experience_requirement_source"],
            "qualification_list_flexible": job["requirements"]["jd_flexible_qualifications"],
            "requirements": job["requirements"]["requirements"],
            "education_requirements": job.get("education_requirements", []),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "file_errors": errors,
        "warnings": warnings,
        "score_comparability": comparability,
        "persisted": False,
        "scoring_methodology_version": candidates[0]["score_methodology_version"] if candidates else UPLOAD_METHOD_VERSION,
        "scoring_formula": candidates[0]["formula"] if candidates else "",
    })


@app.route("/api/case")
def case():
    resume_id = request.args.get("resume_id", "").strip()
    job_id = request.args.get("job_id", "").strip()
    data = load_dashboard()
    matches = [row for row in data["cases"] if row.get("resume_id") == resume_id and (not job_id or row.get("job_id") == job_id)]
    return jsonify(matches)


@app.route("/api/compare")
def compare_candidates():
    job_id = request.args.get("job_id", "").strip()
    resume_a = request.args.get("resume_a", "").strip()
    resume_b = request.args.get("resume_b", "").strip()
    if not job_id or not resume_a or not resume_b:
        return jsonify({"error": "Job ID and both resume IDs are required."}), 400
    if resume_a == resume_b:
        return jsonify({"error": "Choose two different candidates."}), 400
    rows = read_csv(SCREENING / "screening_decision_results.csv")
    found = {resume_id: next((row for row in rows if str(row.get("resume_id", "")).strip() == resume_id and str(row.get("job_id", "")).strip() == job_id), None) for resume_id in (resume_a, resume_b)}
    if any(value is None for value in found.values()):
        missing = [key for key, value in found.items() if value is None]
        return jsonify({"error": "Candidate(s) not found for this job: " + ", ".join(missing)}), 404
    return jsonify({"job_id": job_id, "candidate_a": found[resume_a], "candidate_b": found[resume_b]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
