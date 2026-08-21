import pandas as pd


SELECTION_THRESHOLD = 0.70


def calculate_fairness(input_file, output_file):

    df = pd.read_csv(input_file)

    # Determine whether each candidate is selected
    df["selected"] = (
        df["score"] >= SELECTION_THRESHOLD
    ).astype(int)

    # Calculate selection rate for each group
    summary = (
        df.groupby("group")
        .agg(
            total_candidates=("candidate_id", "count"),
            selected_candidates=("selected", "sum"),
            average_score=("score", "mean")
        )
        .reset_index()
    )

    summary["selection_rate"] = (
        summary["selected_candidates"]
        / summary["total_candidates"]
    )

    reference_group = "Tier_1"

    reference_rate = summary.loc[
        summary["group"] == reference_group,
        "selection_rate"
    ].iloc[0]

    # Disparate Impact Ratio
    summary["disparate_impact_ratio"] = (
        summary["selection_rate"]
        / reference_rate
    )

    # Selection rate difference
    summary["selection_rate_difference"] = (
        summary["selection_rate"]
        - reference_rate
    )

    summary.to_csv(
        output_file,
        index=False
    )

    print("\nFairness Evaluation:")
    print(summary.to_string(index=False))


if __name__ == "__main__":

    calculate_fairness(
        "data/mock_fairness.csv",
        "outputs/fairness_results.csv"
    )