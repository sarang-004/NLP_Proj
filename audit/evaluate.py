import pandas as pd


def evaluate_bias(input_file, output_file):
    df = pd.read_csv(input_file)

    summary = (
        df.groupby("proxy_type")["delta"]
        .agg(
            mean_delta="mean",
            median_delta="median",
            std_delta="std",
            min_delta="min",
            max_delta="max",
            cases="count"
        )
        .reset_index()
    )

    # Count negative and positive changes
    negative_counts = (
        df[df["delta"] < 0]
        .groupby("proxy_type")
        .size()
        .reset_index(name="negative_cases")
    )

    positive_counts = (
        df[df["delta"] > 0]
        .groupby("proxy_type")
        .size()
        .reset_index(name="positive_cases")
    )

    # Combine results
    summary = summary.merge(
        negative_counts,
        on="proxy_type",
        how="left"
    )

    summary = summary.merge(
        positive_counts,
        on="proxy_type",
        how="left"
    )

    # Replace missing values with 0
    summary = summary.fillna(0)

    summary.to_csv(output_file, index=False)

    print("\nBias Evaluation Results:")
    print(summary)


if __name__ == "__main__":
    evaluate_bias(
        "outputs/bias_results.csv",
        "outputs/fairness_summary.csv"
    )