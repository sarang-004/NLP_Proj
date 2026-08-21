import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("outputs")


def generate_report():

    # Load audit results
    bias_df = pd.read_csv(
        OUTPUT_DIR / "bias_results.csv"
    )

    fairness_summary = pd.read_csv(
        OUTPUT_DIR / "fairness_summary.csv"
    )

    context_df = pd.read_csv(
        OUTPUT_DIR / "context_results.csv"
    )

    intersection_df = pd.read_csv(
        OUTPUT_DIR / "intersectionality_results.csv"
    )

    fairness_df = pd.read_csv(
        OUTPUT_DIR / "fairness_results.csv"
    )

    report_path = OUTPUT_DIR / "audit_report.txt"

    with open(report_path, "w", encoding="utf-8") as report:

        report.write("=" * 60 + "\n")
        report.write("AUTOMATED RESUME BIAS AUDIT REPORT\n")
        report.write("=" * 60 + "\n\n")

        # --------------------------------------------------
        # DATASET SUMMARY
        # --------------------------------------------------

        report.write("1. DATASET SUMMARY\n")
        report.write("-" * 40 + "\n")

        total_cases = len(bias_df)
        unique_resumes = bias_df["resume_id"].nunique()

        report.write(
            f"Unique resumes analysed: {unique_resumes}\n"
        )

        report.write(
            f"Counterfactual cases analysed: {total_cases}\n\n"
        )

        # --------------------------------------------------
        # FACTOR ANALYSIS
        # --------------------------------------------------

        report.write("2. CONTEXTUAL FACTOR ANALYSIS\n")
        report.write("-" * 40 + "\n")

        for _, row in fairness_summary.iterrows():

            report.write(
                f"\nFactor: {row['proxy_type']}\n"
            )

            report.write(
                f"Mean delta: {row['mean_delta']:.4f}\n"
            )

            report.write(
                f"Median delta: {row['median_delta']:.4f}\n"
            )

            report.write(
                f"Standard deviation: "
                f"{row['std_delta']:.4f}\n"
            )

            report.write(
                f"Cases: {int(row['cases'])}\n"
            )

            report.write(
                f"Negative cases: "
                f"{int(row['negative_cases'])}\n"
            )

            report.write(
                f"Positive cases: "
                f"{int(row['positive_cases'])}\n"
            )

        # --------------------------------------------------
        # LARGEST EFFECT
        # --------------------------------------------------

        largest_negative = fairness_summary.loc[
            fairness_summary["mean_delta"].idxmin()
        ]

        report.write("\n\n3. LARGEST NEGATIVE EFFECT\n")
        report.write("-" * 40 + "\n")

        report.write(
            f"Factor: {largest_negative['proxy_type']}\n"
        )

        report.write(
            f"Mean delta: "
            f"{largest_negative['mean_delta']:.4f}\n"
        )

        # --------------------------------------------------
        # CONTEXT ANALYSIS
        # --------------------------------------------------

        report.write("\n\n4. CONTEXT ANALYSIS\n")
        report.write("-" * 40 + "\n")

        context_counts = (
            context_df["context_classification"]
            .value_counts()
        )

        for classification, count in context_counts.items():

            report.write(
                f"{classification}: {count}\n"
            )

        # Potentially concerning cases
        unsupported = context_df[
            context_df["context_classification"]
            == "Not Supported"
        ]

        report.write(
            f"\nPotentially unsupported contextual cases: "
            f"{len(unsupported)}\n"
        )

        # --------------------------------------------------
        # INTERSECTIONALITY
        # --------------------------------------------------

        report.write("\n\n5. INTERSECTIONAL ANALYSIS\n")
        report.write("-" * 40 + "\n")

        avg_expected = (
            intersection_df["expected_effect"].mean()
        )

        avg_actual = (
            intersection_df[
                "actual_intersection_effect"
            ].mean()
        )

        avg_interaction = (
            intersection_df[
                "interaction_effect"
            ].mean()
        )

        report.write(
            f"Average expected effect: "
            f"{avg_expected:.4f}\n"
        )

        report.write(
            f"Average actual intersection effect: "
            f"{avg_actual:.4f}\n"
        )

        report.write(
            f"Average interaction effect: "
            f"{avg_interaction:.4f}\n"
        )

        # --------------------------------------------------
        # FAIRNESS
        # --------------------------------------------------

        report.write("\n\n6. GROUP FAIRNESS\n")
        report.write("-" * 40 + "\n")

        for _, row in fairness_df.iterrows():

            report.write(
                f"\nGroup: {row['group']}\n"
            )

            report.write(
                f"Candidates: "
                f"{int(row['total_candidates'])}\n"
            )

            report.write(
                f"Selected: "
                f"{int(row['selected_candidates'])}\n"
            )

            report.write(
                f"Average score: "
                f"{row['average_score']:.4f}\n"
            )

            report.write(
                f"Selection rate: "
                f"{row['selection_rate']:.4f}\n"
            )

            report.write(
                f"Disparate impact ratio: "
                f"{row['disparate_impact_ratio']:.4f}\n"
            )

        # --------------------------------------------------
        # VISUALIZATIONS
        # --------------------------------------------------

        report.write("\n\n7. GENERATED VISUALIZATIONS\n")
        report.write("-" * 40 + "\n")

        plots = [
            "mean_delta.png",
            "base_vs_variant.png",
            "intersectionality.png",
            "selection_rates.png"
        ]

        for plot in plots:

            report.write(
                f"outputs/plots/{plot}\n"
            )

        # --------------------------------------------------
        # FINAL NOTE
        # --------------------------------------------------

        report.write("\n\n8. INTERPRETATION NOTE\n")
        report.write("-" * 40 + "\n")

        report.write(
            "A score difference alone does not establish "
            "unfair discrimination. Contextual factors "
            "should be interpreted against job requirements "
            "and relevant candidate qualifications. "
            "Unsupported score differences should be "
            "flagged for further investigation.\n"
        )

    print(
        f"\nAudit report generated successfully:\n"
        f"{report_path}"
    )


if __name__ == "__main__":
    generate_report()