import pandas as pd


def calculate_intersectionality(input_file, output_file):
    df = pd.read_csv(input_file)

    # Factors that are considered individual effects
    individual_factors = [
        "college",
        "city",
        "name"
    ]

    results = []

    # Process each resume separately
    for resume_id, group in df.groupby("resume_id"):

        # Get individual factor deltas
        individual_data = group[
            group["proxy_type"].isin(individual_factors)
        ]

        # Sum of individual effects
        expected_effect = individual_data["delta"].sum()

        # Get actual intersectional effect
        intersection_data = group[
            group["proxy_type"] == "intersection"
        ]

        if intersection_data.empty:
            continue

        actual_effect = intersection_data["delta"].iloc[0]

        # Calculate additional interaction effect
        interaction_effect = (
            actual_effect - expected_effect
        )

        results.append({
            "resume_id": resume_id,
            "expected_effect": expected_effect,
            "actual_intersection_effect": actual_effect,
            "interaction_effect": interaction_effect
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(output_file, index=False)

    print("\nIntersectionality Results:")
    print(results_df)

    # Overall summary
    if not results_df.empty:
        print("\nOverall Summary:")
        print(
            "Average Expected Effect:",
            results_df["expected_effect"].mean()
        )

        print(
            "Average Actual Intersection Effect:",
            results_df["actual_intersection_effect"].mean()
        )

        print(
            "Average Interaction Effect:",
            results_df["interaction_effect"].mean()
        )


if __name__ == "__main__":
    calculate_intersectionality(
        "data/mock_intersectionality.csv",
        "outputs/intersectionality_results.csv"
    )