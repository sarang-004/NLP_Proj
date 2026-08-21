import pandas as pd


def calculate_deltas(input_file, output_file):
    df = pd.read_csv(input_file)

    # Separate original/base resumes
    base_df = df[df["proxy_type"] == "base"][
        ["resume_id", "score"]
    ].rename(columns={"score": "base_score"})

    # Get all counterfactual variants
    variants_df = df[df["proxy_type"] != "base"].copy()

    # Attach the corresponding base score
    variants_df = variants_df.merge(
        base_df,
        on="resume_id",
        how="left"
    )

    # Calculate delta
    variants_df["delta"] = (
        variants_df["score"] - variants_df["base_score"]
    )

    # Rename score for clarity
    variants_df = variants_df.rename(
        columns={"score": "variant_score"}
    )

    # Select useful columns
    result = variants_df[
        [
            "resume_id",
            "variant_id",
            "proxy_type",
            "proxy_value",
            "base_score",
            "variant_score",
            "delta"
        ]
    ]

    result.to_csv(output_file, index=False)

    print("Delta calculation completed.")
    print(result)


if __name__ == "__main__":
    calculate_deltas(
        "data/mock_scores.csv",
        "outputs/bias_results.csv"
    )