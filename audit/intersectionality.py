import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "screening_output_final" / "screening_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PAIRS = {
    "college+city": ("college", "city"),
    "college+name": ("college", "name"),
    "name+city": ("name", "city"),
    "degree+course": ("degree", "course"),
}


def calculate_intersectionality(input_file, output_file):
    df = pd.read_csv(input_file)

    required = {
        "resume_id", "job_id", "proxy_type",
        "base_score", "variant_score"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["base_score"] = pd.to_numeric(df["base_score"], errors="coerce")
    df["variant_score"] = pd.to_numeric(df["variant_score"], errors="coerce")

    df["delta"] = df["variant_score"] - df["base_score"]

    # Ensure each resume/job/factor has one unambiguous score change.
    key_cols = ["resume_id", "job_id", "proxy_type"]

    if df.duplicated(key_cols, keep=False).any():
        raise ValueError(
            "Duplicate resume/job/proxy combinations found. "
            "Resolve them before calculating interactions."
        )

    results = []

    for joint_type, (factor_a, factor_b) in PAIRS.items():
        joint = df[df["proxy_type"] == joint_type].copy()

        individual_a = df[df["proxy_type"] == factor_a][
            ["resume_id", "job_id", "delta"]
        ].rename(columns={"delta": "delta_a"})

        individual_b = df[df["proxy_type"] == factor_b][
            ["resume_id", "job_id", "delta"]
        ].rename(columns={"delta": "delta_b"})

        matched = joint.merge(
            individual_a,
            on=["resume_id", "job_id"],
            how="left",
            validate="one_to_one",
        ).merge(
            individual_b,
            on=["resume_id", "job_id"],
            how="left",
            validate="one_to_one",
        )

        matched["match_status"] = "matched"
        matched.loc[
            matched[["delta_a", "delta_b"]].isna().any(axis=1),
            "match_status",
        ] = "missing_individual_effect"

        matched["expected_additive_delta"] = (
            matched["delta_a"] + matched["delta_b"]
        )

        matched["interaction_effect"] = (
            matched["delta"] - matched["expected_additive_delta"]
        )

        matched["factor_a"] = factor_a
        matched["factor_b"] = factor_b

        results.append(matched)

    columns = [
        "resume_id", "job_id", "variant_id", "proxy_type",
        "factor_a", "factor_b", "base_score", "variant_score",
        "actual_joint_delta", "delta_a", "delta_b",
        "expected_additive_delta", "interaction_effect",
        "match_status",
    ]

    if results:
        detailed = pd.concat(results, ignore_index=True)

        # Rename before selecting columns.
        detailed = detailed.rename(
            columns={"delta": "actual_joint_delta"}
        )

        detailed = detailed[columns]
    else:
        detailed = pd.DataFrame(columns=columns)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(output_file, index=False)

    # Summarize only cases with both individual effects available.
    valid = detailed[
        detailed["match_status"] == "matched"
    ].copy()

    summary_columns = [
        "proxy_type",
        "cases",
        "mean_expected_delta",
        "mean_actual_joint_delta",
        "mean_interaction_effect",
        "median_interaction_effect",
        "mean_absolute_interaction",
    ]

    if not valid.empty:
        summary = (
            valid.groupby("proxy_type")
            .agg(
                cases=("interaction_effect", "count"),
                mean_expected_delta=("expected_additive_delta", "mean"),
                mean_actual_joint_delta=("actual_joint_delta", "mean"),
                mean_interaction_effect=("interaction_effect", "mean"),
                median_interaction_effect=("interaction_effect", "median"),
                mean_absolute_interaction=(
                    "interaction_effect",
                    lambda x: x.abs().mean(),
                ),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(columns=summary_columns)

    summary_file = output_file.parent / "intersectionality_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"Detailed results saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Joint cases: {len(detailed)}")
    print(f"Matched cases: {len(valid)}")
    print(f"Cases missing individual effects: {len(detailed) - len(valid)}")

    if not summary.empty:
        print("\nInteraction summary:")
        print(summary.to_string(index=False))
    else:
        print("\nNo valid matched cases available for the summary.")


if __name__ == "__main__":
    calculate_intersectionality(
        INPUT_FILE,
        OUTPUT_DIR / "intersectionality_results.csv",
    )
