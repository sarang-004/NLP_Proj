import pandas as pd


def calculate_deltas(input_file, output_file):

    df = pd.read_csv(input_file)

    version_columns = {"base_score_methodology_version", "variant_score_methodology_version"}
    if version_columns.issubset(df.columns):
        incompatible = df["base_score_methodology_version"].astype(str) != df["variant_score_methodology_version"].astype(str)
        if incompatible.any():
            raise ValueError("Cannot calculate counterfactual deltas across scoring methodology versions.")
        df["score_methodology_version"] = df["base_score_methodology_version"]
    else:
        # Existing validated outputs predate score versioning; retain them as
        # explicitly legacy/unversioned instead of relabeling them as v2.
        df["score_methodology_version"] = "legacy_unversioned"

    # Calculate score difference
    df["delta"] = (
        df["variant_score"] - df["base_score"]
    )

    # Keep useful columns
    result = df[
        [
            "resume_id",
            "job_id",
            "variant_id",
            "proxy_type",
            "proxy_value",
            "base_score",
            "variant_score",
            "score_methodology_version",
            "delta"
        ]
    ].copy()

    # Save results
    result.to_csv(
        output_file,
        index=False
    )

    print("Delta calculation completed.")
    print(f"Total counterfactual cases: {len(result)}")
    print("\nFirst 10 results:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":

    calculate_deltas(
        "screening_output_final/screening_results.csv",
        "outputs/bias_results.csv"
    )
