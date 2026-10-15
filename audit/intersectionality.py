"""Transformation-aware intersectional score analysis."""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "screening_output_final" / "screening_results_detailed.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PAIRS = {
    "college+city": ("college", "city"),
    "college+name": ("college", "name"),
    "name+city": ("name", "city"),
    "degree+course": ("degree", "course"),
}


def _parts(value):
    return [str(item).strip() for item in str(value or "").split("|")]


def _same(left, right):
    return str(left or "").strip().casefold() == str(right or "").strip().casefold()


def validate_intersectionality(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate interactions only when individual transformation values match.

    A match requires the same candidate and job plus equal original and variant
    values for each factor. Score changes are always recalculated from scores.
    """
    required = {"resume_id", "job_id", "variant_id", "proxy_type", "base_score", "variant_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.duplicated(["resume_id", "job_id", "variant_id"]).any():
        raise ValueError("Duplicate resume/job/variant keys in screening results.")

    df = frame.copy()
    for field in ("base_score", "variant_score"):
        df[field] = pd.to_numeric(df[field], errors="coerce")
    df["_delta"] = df["variant_score"] - df["base_score"]
    source_lookup = {}
    for _, row in df.iterrows():
        key = (str(row["resume_id"]), str(row["job_id"]), str(row["proxy_type"]))
        if key in source_lookup:
            raise ValueError(f"Duplicate candidate/job/factor result: {key}")
        source_lookup[key] = row

    records = []
    for _, joint in df[df["proxy_type"].isin(PAIRS)].iterrows():
        factor_a, factor_b = PAIRS[joint["proxy_type"]]
        joint_original = _parts(joint.get("original_value", ""))
        joint_variant = _parts(joint.get("counterfactual_value", joint.get("proxy_value", "")))
        individual = [
            source_lookup.get((str(joint["resume_id"]), str(joint["job_id"]), factor_a)),
            source_lookup.get((str(joint["resume_id"]), str(joint["job_id"]), factor_b)),
        ]
        status = "matched"
        version_field = "base_score_methodology_version"
        joint_version = str(joint.get(version_field, "legacy_unversioned") or "legacy_unversioned")
        individual_versions = [str(item.get(version_field, "legacy_unversioned") or "legacy_unversioned") if item is not None else "" for item in individual]
        all_rows = [joint, *[item for item in individual if item is not None]]
        internal_version_mismatch = any(
            "base_score_methodology_version" in item.index
            and "variant_score_methodology_version" in item.index
            and str(item.get("base_score_methodology_version", "")) != str(item.get("variant_score_methodology_version", ""))
            for item in all_rows
        )
        if any(item is None for item in individual):
            status = "missing_individual_effect"
        elif internal_version_mismatch or any(version != joint_version for version in individual_versions):
            status = "scoring_version_mismatch"
        elif len(joint_original) != 2 or len(joint_variant) != 2:
            status = "missing_joint_transformation_values"
        else:
            for i, item in enumerate(individual):
                if not _same(item.get("original_value", ""), joint_original[i]) or not _same(
                    item.get("counterfactual_value", item.get("proxy_value", "")), joint_variant[i]
                ):
                    status = "transformation_mismatch"
                    break

        delta_a = individual[0]["_delta"] if individual[0] is not None else float("nan")
        delta_b = individual[1]["_delta"] if individual[1] is not None else float("nan")
        actual = joint["_delta"]
        matched = status == "matched" and pd.notna(delta_a) and pd.notna(delta_b) and pd.notna(actual)
        expected = delta_a + delta_b if matched else float("nan")
        records.append({
            "resume_id": joint["resume_id"],
            "job_id": joint["job_id"],
            "variant_id": joint["variant_id"],
            "proxy_type": joint["proxy_type"],
            "factor_a": factor_a,
            "factor_b": factor_b,
            "factor_a_original_value": joint_original[0] if len(joint_original) == 2 else "",
            "factor_a_variant_value": joint_variant[0] if len(joint_variant) == 2 else "",
            "factor_b_original_value": joint_original[1] if len(joint_original) == 2 else "",
            "factor_b_variant_value": joint_variant[1] if len(joint_variant) == 2 else "",
            "base_score": joint["base_score"],
            "variant_score": joint["variant_score"],
            "actual_joint_delta": actual,
            "delta_a": delta_a if matched else float("nan"),
            "delta_b": delta_b if matched else float("nan"),
            "expected_additive_delta": expected,
            "interaction_effect": actual - expected if matched else float("nan"),
            "score_methodology_version": joint_version,
            "match_status": "matched" if matched else status,
        })
    return pd.DataFrame(records)


def summarize_intersectionality(detailed: pd.DataFrame) -> pd.DataFrame:
    columns = ["proxy_type", "cases", "mean_expected_delta", "mean_actual_joint_delta", "mean_interaction_effect", "median_interaction_effect", "mean_absolute_interaction"]
    if detailed.empty:
        return pd.DataFrame(columns=columns)
    valid = detailed[detailed["match_status"] == "matched"]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    return valid.groupby("proxy_type").agg(
        cases=("interaction_effect", "count"),
        mean_expected_delta=("expected_additive_delta", "mean"),
        mean_actual_joint_delta=("actual_joint_delta", "mean"),
        mean_interaction_effect=("interaction_effect", "mean"),
        median_interaction_effect=("interaction_effect", "median"),
        mean_absolute_interaction=("interaction_effect", lambda values: values.abs().mean()),
    ).reset_index()


def calculate_intersectionality(input_file=INPUT_FILE, output_file=None):
    """Explicit CLI pipeline; dashboard calls the pure validator directly."""
    detailed = validate_intersectionality(pd.read_csv(input_file, keep_default_na=False))
    output = Path(output_file) if output_file else OUTPUT_DIR / "intersectionality_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(output, index=False)
    summary = summarize_intersectionality(detailed)
    summary.to_csv(output.parent / "intersectionality_summary.csv", index=False)
    print(f"Joint cases: {len(detailed)}")
    print(f"Transformation-compatible cases: {int(detailed['match_status'].eq('matched').sum()) if not detailed.empty else 0}")
    print(f"Cases requiring review: {int(detailed['match_status'].ne('matched').sum()) if not detailed.empty else 0}")
    return detailed, summary


if __name__ == "__main__":
    calculate_intersectionality()
