import pandas as pd


def calculate_deltas(input_file, output_file):

    df = pd.read_csv(input_file)

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