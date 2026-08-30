import pandas as pd
import numpy as np


def bootstrap_ci(values, iterations=5000, confidence=0.95):

    values = np.array(values)

    bootstrap_means = []

    for _ in range(iterations):
        sample = np.random.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_means.append(
            np.mean(sample)
        )

    alpha = 1 - confidence

    lower = np.percentile(
        bootstrap_means,
        100 * (alpha / 2)
    )

    upper = np.percentile(
        bootstrap_means,
        100 * (1 - alpha / 2)
    )

    return lower, upper


def evaluate_bias(input_file, output_file):

    df = pd.read_csv(input_file)

    results = []

    for proxy_type, group in df.groupby("proxy_type"):

        deltas = group["delta"].dropna()

        mean_delta = deltas.mean()

        ci_lower, ci_upper = bootstrap_ci(
            deltas
        )

        results.append({
            "proxy_type": proxy_type,
            "mean_delta": mean_delta,
            "median_delta": deltas.median(),
            "std_delta": deltas.std(),
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "min_delta": deltas.min(),
            "max_delta": deltas.max(),
            "cases": len(deltas),
            "negative_cases": (deltas < 0).sum(),
            "positive_cases": (deltas > 0).sum()
        })

    summary = pd.DataFrame(results)

    summary.to_csv(
        output_file,
        index=False
    )

    print("\nBias Evaluation Results:")
    print(
        summary.to_string(index=False)
    )


if __name__ == "__main__":

    evaluate_bias(
        "outputs/bias_results.csv",
        "outputs/fairness_summary.csv"
    )