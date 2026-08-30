import pandas as pd


def calculate_intersectionality(input_file, output_file):

    df = pd.read_csv(input_file)

    # Intersectional combinations available in the real dataset
    combinations = [
        ("college+city", ["college", "city"]),
        ("college+experience", ["college", "experience"]),
        ("college+name", ["college", "name"]),
        ("course+experience", ["course", "experience"]),
        ("degree+age", ["degree", "age"]),
        ("degree+course", ["degree", "course"]),
        ("name+age", ["name", "age"]),
        ("name+city", ["name", "city"])
    ]

    results = []

    for intersection_name, factors in combinations:

        intersection_data = df[
            df["proxy_type"] == intersection_name
        ]

        if intersection_data.empty:
            continue

        # Calculate individual effects for the same resumes
        individual_data = df[
            df["proxy_type"].isin(factors)
        ]

        individual_means = (
            individual_data
            .groupby("resume_id")["delta"]
            .mean()
        )

        for _, row in intersection_data.iterrows():

            resume_id = row["resume_id"]

            actual_effect = row["delta"]

            # Get individual effects for this resume
            resume_individual = individual_data[
                individual_data["resume_id"] == resume_id
            ]

            if resume_individual.empty:
                continue

            individual_effects = (
                resume_individual
                .groupby("proxy_type")["delta"]
                .mean()
            )

            expected_effect = sum(
                individual_effects.get(
                    factor,
                    0
                )
                for factor in factors
            )

            interaction_effect = (
                actual_effect - expected_effect
            )

            results.append({
                "resume_id": resume_id,
                "intersection_type": intersection_name,
                "expected_effect": expected_effect,
                "actual_intersection_effect": actual_effect,
                "interaction_effect": interaction_effect
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nIntersectionality Results:")

    if results_df.empty:
        print("No intersectional results found.")
        return

    print(
        results_df.head(20)
        .to_string(index=False)
    )

    print("\nOverall Summary:")

    summary = (
        results_df
        .groupby("intersection_type")
        .agg(
            mean_expected_effect=(
                "expected_effect",
                "mean"
            ),
            mean_actual_effect=(
                "actual_intersection_effect",
                "mean"
            ),
            mean_interaction_effect=(
                "interaction_effect",
                "mean"
            ),
            cases=(
                "interaction_effect",
                "count"
            )
        )
        .reset_index()
    )

    print(
        summary.to_string(index=False)
    )

    summary.to_csv(
        "outputs/intersectionality_summary.csv",
        index=False
    )


if __name__ == "__main__":

    calculate_intersectionality(
        "outputs/bias_results.csv",
        "outputs/intersectionality_results.csv"
    )