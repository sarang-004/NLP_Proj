"""Generate a consolidated report from the validated project outputs.

This module summarizes recorded results only. It does not infer hiring
outcomes or treat score sensitivity as proof of discrimination.
"""

from pathlib import Path
import math

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SCREENING_DIR = PROJECT_ROOT / "screening_output_final"


def _read(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _number(value, digits=4):
    try:
        value = float(value)
        return f"{value:.{digits}f}" if math.isfinite(value) else "unavailable"
    except (TypeError, ValueError):
        return "unavailable"


def generate_report(output_path=None):
    """Write a reproducible text report based on current CSV outputs."""
    detailed = pd.read_csv(SCREENING_DIR / "screening_results_detailed.csv") if (SCREENING_DIR / "screening_results_detailed.csv").exists() else pd.DataFrame()
    base = pd.read_csv(SCREENING_DIR / "screening_decision_results.csv") if (SCREENING_DIR / "screening_decision_results.csv").exists() else pd.DataFrame()
    unified = _read("unified_audit.csv")
    from app import load_dashboard
    dashboard = load_dashboard()
    context = pd.DataFrame(dashboard["cases"])
    fairness = pd.DataFrame(dashboard["fairness"])
    sensitivity = pd.DataFrame(dashboard["fairness_sensitivity"])
    intersections = pd.DataFrame(dashboard["intersectionality"])
    intersection_summary = pd.DataFrame(dashboard["intersectionality_summary"])
    validation = pd.read_csv(SCREENING_DIR / "counterfactual_scoring_validation.csv") if (SCREENING_DIR / "counterfactual_scoring_validation.csv").exists() else pd.DataFrame()
    explanation_validation_path = OUTPUT_DIR / "explainability" / "explainability_validation.csv"
    explanation_validation = pd.read_csv(explanation_validation_path) if explanation_validation_path.exists() else pd.DataFrame()

    lines = [
        "AUTOMATED RESUME SCREENING AND COUNTERFACTUAL AUDIT",
        "=" * 58,
        "",
        "Scope and interpretation",
        "This report summarizes stored screening and audit outputs. A score change or decision flip measures model sensitivity under a tested transformation; it does not establish discrimination, job suitability, or predictive validity.",
        "",
        "Dataset and screening methodology",
        f"Base candidate-job results: {len(base)}",
        f"Detailed counterfactual screening results: {len(detailed)}",
        f"Unified audit rows: {len(unified)}",
        f"Unique audit cases including unmatched supplemental rows: {dashboard['metrics']['total_cases']}; verified: {dashboard['metrics']['verified_cases']}; unmatched: {dashboard['metrics']['unmatched_cases']}; invalid: {dashboard['metrics']['invalid_cases']}; conflicting supplemental scores: {dashboard['metrics']['supplemental_conflicts']}.",
        "Scoring components recorded by the pipeline: skills, experience, education/course, projects, and semantic similarity.",
        "The audit consumes stored scores and does not change the screening formula or ranking.",
        "",
        "Factor explanations",
    ]
    if not explanation_validation.empty:
        row = explanation_validation.iloc[0]
        lines += [f"Base explanations: {row.get('explanations_generated', 'unavailable')} / {row.get('base_candidates', 'unavailable')}; missing screening details: {row.get('missing_screening_details', 'unavailable')}."]
    else:
        lines += ["Explanation validation output is unavailable."]

    lines += ["", "Counterfactual results"]
    if not unified.empty and "proxy_type" in unified:
        for proxy, group in unified.groupby("proxy_type", dropna=False):
            delta_col = "score_delta" if "score_delta" in group else "delta"
            deltas = pd.to_numeric(group.get(delta_col), errors="coerce")
            lines.append(f"{proxy}: n={deltas.notna().sum()}, mean delta={_number(deltas.mean())}, median delta={_number(deltas.median())}.")
    else:
        lines.append("Unified counterfactual audit output is unavailable.")
    if not validation.empty:
        checked = int(validation["status"].astype(str).str.lower().eq("passed").sum()) if "status" in validation else 0
        lines.append(f"Single-attribute validation: {checked} passed rows out of {len(validation)} records in the validation file.")
    covered = set(detailed.get("proxy_type", pd.Series(dtype=str)).astype(str))
    for attribute in ("name", "age", "college", "city", "course", "degree", "experience", "employment_gap_months"):
        lines.append(f"{attribute} counterfactual coverage: {'scored' if attribute in covered or any(f'{attribute}+' in item or f'+{attribute}' in item for item in covered) else 'not present in current authoritative detailed scores'}.")

    lines += ["", "Decision flips and threshold sensitivity"]
    lines.append(f"Verified direct-proxy cases: {dashboard['metrics']['direct_proxy_tests']}.")
    lines.append(f"Cases flipping at one or more supported thresholds: {dashboard['metrics']['decision_flips']}.")
    for threshold, count in dashboard["thresholds"].items():
        lines.append(f"{threshold}: {count} verified threshold flips.")

    lines += ["", "Contextual findings"]
    if not context.empty and "context_classification" in context:
        counts = context["context_classification"].fillna("Unavailable").value_counts(dropna=False)
        lines.extend(f"{label}: {count}" for label, count in counts.items())
        lines.append("Context labels are evidence-based audit classifications, not legal determinations.")
    else:
        lines.append("Context output is unavailable.")

    lines += ["", "Intersectional results"]
    if not intersections.empty:
        status = intersections.get("match_status", pd.Series(dtype=str)).value_counts()
        lines.append(f"Joint cases: {len(intersections)}; matched individual-effect comparisons: {int(status.get('matched', 0))}; unmatched: {len(intersections) - int(status.get('matched', 0))}.")
        if not intersection_summary.empty:
            for _, row in intersection_summary.iterrows():
                lines.append(f"{row.get('proxy_type', 'unknown')}: n={row.get('cases', 'unavailable')}, mean interaction={_number(row.get('mean_interaction_effect'))}, mean absolute interaction={_number(row.get('mean_absolute_interaction'))}.")
        lines.append("Transformation compatibility was checked against original and variant values. Interaction effect is the deviation from an additive expectation under the tested scoring system; it is not proof of discrimination.")
    else:
        lines.append("Intersectionality output is unavailable.")

    lines += ["", "Fairness metrics and limitations"]
    if not fairness.empty:
        lines.append(f"Recorded group-threshold metric rows: {len(fairness)}.")
        if {"factor", "group", "total_candidates", "selection_rate"}.issubset(fairness.columns):
            for (factor, group), rows in fairness.groupby(["factor", "group"], dropna=False):
                row = rows.iloc[0]
                lines.append(f"{factor} / {group}: n={row.get('total_candidates', 'unavailable')}, selection rate={_number(row.get('selection_rate'))}, disparate impact ratio={_number(row.get('disparate_impact_ratio'))}.")
    else:
        lines.append("Fairness group metrics are unavailable.")
    if not sensitivity.empty:
        lines.append(f"Threshold sensitivity rows: {len(sensitivity)} at the thresholds recorded in the source output.")
    lines.append("The dataset contains no ground-truth hiring outcomes. Group metrics describe these records and thresholds only; sample size, representation, and missing or proxy attributes limit interpretation.")

    lines += ["", "Data quality, uncertainty, and responsible use"]
    if not unified.empty and {"resume_id", "job_id", "variant_id"}.issubset(unified.columns):
        lines.append(f"Unified audit rows with duplicate resume/job/variant keys: {int(unified.duplicated(['resume_id', 'job_id', 'variant_id']).sum())}.")
    lines.append("Do not use protected or proxy attributes as hiring criteria. Review cases with unsupported context, missing evidence, or consequential score changes with qualified human reviewers.")
    lines.append("No LLM-generated assessments are included; this report contains measured outputs and their limitations only.")

    destination = Path(output_path) if output_path else OUTPUT_DIR / "audit_report.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(f"Report generated: {generate_report()}")
