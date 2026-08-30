import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Create output folder if it does not exist
OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_mean_delta():
    """
    Plot average score change for each contextual factor.
    """

    df = pd.read_csv("outputs/fairness_summary.csv")

    plt.figure(figsize=(9, 5))

    plt.bar(
        df["proxy_type"],
        df["mean_delta"]
    )

    plt.axhline(
        y=0,
        linewidth=1
    )

    plt.title("Mean Score Delta by Contextual Factor")
    plt.xlabel("Contextual Factor")
    plt.ylabel("Mean Score Delta")

    plt.xticks(rotation=30)
    plt.tight_layout()

    path = OUTPUT_DIR / "mean_delta.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Created: {path}")


def plot_base_vs_variant():
    """
    Plot base and variant scores for each counterfactual.
    """

    df = pd.read_csv("outputs/bias_results.csv")

    plt.figure(figsize=(10, 6))

    x = range(len(df))

    plt.plot(
        x,
        df["base_score"],
        marker="o",
        label="Base Score"
    )

    plt.plot(
        x,
        df["variant_score"],
        marker="o",
        label="Variant Score"
    )

    plt.title("Base Score vs Counterfactual Variant Score")
    plt.xlabel("Counterfactual Case")
    plt.ylabel("Score")

    plt.xticks(
        x,
        df["proxy_type"],
        rotation=45
    )

    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "base_vs_variant.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Created: {path}")


def plot_intersectionality():
    """
    Compare expected additive effect with
    actual intersectional effect.
    """

    df = pd.read_csv(
        "outputs/intersectionality_results.csv"
    )

    plt.figure(figsize=(9, 5))

    x = range(len(df))

    plt.plot(
        x,
        df["expected_effect"],
        marker="o",
        label="Expected Effect"
    )

    plt.plot(
        x,
        df["actual_intersection_effect"],
        marker="o",
        label="Actual Intersection Effect"
    )

    plt.title(
        "Expected vs Actual Intersectional Effect"
    )

    plt.xlabel("Resume")
    plt.ylabel("Score Effect")

    plt.xticks(
        x,
        df["resume_id"]
    )

    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "intersectionality.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Created: {path}")


def plot_selection_rates():
    """
    Plot selection rates across groups.
    """

    df = pd.read_csv(
        "outputs/fairness_results.csv"
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        df["group"],
        df["selection_rate"]
    )

    plt.title("Selection Rate by Group")
    plt.xlabel("Group")
    plt.ylabel("Selection Rate")

    plt.ylim(0, 1)

    plt.tight_layout()

    path = OUTPUT_DIR / "selection_rates.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Created: {path}")


def main():

    print("\nGenerating audit visualizations...\n")

    plot_mean_delta()
    plot_base_vs_variant()
    plot_intersectionality()
    plot_selection_rates()

    print("\nAll visualizations generated successfully.")
    print(f"Location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()