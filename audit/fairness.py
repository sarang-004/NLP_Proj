import pandas as pd


SCREENING_FILE = "screening_output_final/screening_results.csv"
RESUME_FILE = "final/final_resumes.csv"
COLLEGE_FILE = "intermediate/colleges.csv"
CITY_FILE = "intermediate/cities.csv"

OUTPUT_FILE = "outputs/fairness_results.csv"
SENSITIVITY_FILE = "outputs/fairness_threshold_sensitivity.csv"

THRESHOLDS = [0.55, 0.60, 0.65]


def normalize_text(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(".", "")
        .replace(",", "")
    )


def calculate_group_metrics(df, factor):

    summary = (
        df.dropna(subset=["group"])
        .groupby("group", observed=True)
        .agg(
            total_candidates=("resume_id", "nunique"),
            selected_candidates=("selected", "sum"),
            average_score=("score", "mean")
        )
        .reset_index()
    )

    summary["factor"] = factor

    summary["selection_rate"] = (
        summary["selected_candidates"]
        / summary["total_candidates"]
    )

    if factor in ["college_tier", "city_tier"]:

        tier1 = summary[
            summary["group"].astype(str) == "1"
        ]

        if not tier1.empty:
            reference_rate = tier1[
                "selection_rate"
            ].iloc[0]
        else:
            reference_rate = summary[
                "selection_rate"
            ].max()

    else:

        reference_rate = summary[
            "selection_rate"
        ].max()

    if reference_rate > 0:

        summary["disparate_impact_ratio"] = (
            summary["selection_rate"]
            / reference_rate
        )

    else:

        summary["disparate_impact_ratio"] = 0

    summary["selection_rate_difference"] = (
        summary["selection_rate"]
        - reference_rate
    )

    return summary[
        [
            "factor",
            "group",
            "total_candidates",
            "selected_candidates",
            "average_score",
            "selection_rate",
            "disparate_impact_ratio",
            "selection_rate_difference"
        ]
    ]


def prepare_data():

    print("Loading files...")

    screening = pd.read_csv(SCREENING_FILE)
    resumes = pd.read_csv(RESUME_FILE)
    colleges = pd.read_csv(COLLEGE_FILE)
    cities = pd.read_csv(CITY_FILE)

    # --------------------------------------------------
    # College mapping
    # --------------------------------------------------

    college_aliases = {
        "indian institute of technology bombay": "iit bombay",
        "indian institute of technology delhi": "iit delhi",
        "indian institute of technology madras": "iit madras",
        "indian institute of technology kanpur": "iit kanpur",
        "indian institute of technology kharagpur": "iit kharagpur",
        "indian institute of technology hyderabad": "iit hyderabad",
        "indian institute of technology roorkee": "iit roorkee",
        "indian institute of technology guwahati": "iit guwahati",
        "indian institute of science bangalore": "iisc bangalore",
        "nit tiruchirappalli": "nit trichy",
        "nit trichy": "nit trichy",
        "nit surathkal": "nit surathkal",
        "nit warangal": "nit warangal",
        "nit rourkela": "nit rourkela"
    }

    colleges["college_key"] = (
        colleges["college"]
        .apply(normalize_text)
    )

    resumes["college_key"] = (
        resumes["college"]
        .apply(normalize_text)
    )

    resumes["college_key"] = (
        resumes["college_key"]
        .replace(college_aliases)
    )

    college_map = dict(
        zip(
            colleges["college_key"],
            colleges["college_tier"]
        )
    )

    resumes["college_tier"] = (
        resumes["college_key"]
        .map(college_map)
    )

    # --------------------------------------------------
    # City mapping
    # --------------------------------------------------

    city_aliases = {
        "bangalore": "bengaluru",
        "new delhi": "delhi",
        "trivandrum": "thiruvananthapuram"
    }

    cities["city_key"] = (
        cities["city"]
        .apply(normalize_text)
    )

    resumes["city_key"] = (
        resumes["city"]
        .apply(normalize_text)
    )

    resumes["city_key"] = (
        resumes["city_key"]
        .replace(city_aliases)
    )

    city_map = dict(
        zip(
            cities["city_key"],
            cities["city_tier"]
        )
    )

    resumes["city_tier"] = (
        resumes["city_key"]
        .map(city_map)
    )

    # --------------------------------------------------
    # Base screening records
    # --------------------------------------------------

    base = screening[
        screening["variant_id"]
        .astype(str)
        .str.contains("_BASE")
        |
        (screening["proxy_type"] == "base")
    ].copy()

    if base.empty:

        base = (
            screening
            .sort_values("variant_id")
            .drop_duplicates(
                subset=["resume_id", "job_id"]
            )
            .copy()
        )

    base["score"] = base["base_score"]

    # --------------------------------------------------
    # Merge resume information
    # --------------------------------------------------

    context_columns = [
        "resume_id",
        "college_tier",
        "city_tier",
        "degree_raw",
        "course_final",
        "age_final",
        "total_experience_years_final"
    ]

    available_columns = [
        column
        for column in context_columns
        if column in resumes.columns
    ]

    base = base.merge(
        resumes[available_columns],
        on="resume_id",
        how="left"
    )

    return base


def calculate_fairness_for_threshold(base, threshold):

    df = base.copy()

    df["selected"] = (
        df["score"] >= threshold
    ).astype(int)

    results = []

    # College tier
    if "college_tier" in df.columns:

        temp = df.copy()
        temp["group"] = temp["college_tier"]

        results.append(
            calculate_group_metrics(
                temp,
                "college_tier"
            )
        )

    # City tier
    if "city_tier" in df.columns:

        temp = df.copy()
        temp["group"] = temp["city_tier"]

        results.append(
            calculate_group_metrics(
                temp,
                "city_tier"
            )
        )

    # Degree
    if "degree_raw" in df.columns:

        temp = df.copy()
        temp["group"] = temp["degree_raw"]

        results.append(
            calculate_group_metrics(
                temp,
                "degree"
            )
        )

    # Course
    if "course_final" in df.columns:

        temp = df.copy()
        temp["group"] = temp["course_final"]

        results.append(
            calculate_group_metrics(
                temp,
                "course"
            )
        )

    # Age
    if "age_final" in df.columns:

        temp = df.copy()

        temp["group"] = pd.cut(
            pd.to_numeric(
                temp["age_final"],
                errors="coerce"
            ),
            bins=[0, 24, 29, 34, 39, 100],
            labels=[
                "<=24",
                "25-29",
                "30-34",
                "35-39",
                "40+"
            ]
        )

        results.append(
            calculate_group_metrics(
                temp,
                "age_group"
            )
        )

    # Experience
    if "total_experience_years_final" in df.columns:

        temp = df.copy()

        temp["group"] = pd.cut(
            pd.to_numeric(
                temp["total_experience_years_final"],
                errors="coerce"
            ),
            bins=[-1, 1, 3, 5, 10, 100],
            labels=[
                "<=1",
                "1-3",
                "3-5",
                "5-10",
                "10+"
            ]
        )

        results.append(
            calculate_group_metrics(
                temp,
                "experience_group"
            )
        )

    if not results:
        return pd.DataFrame()

    final_results = pd.concat(
        results,
        ignore_index=True
    )

    final_results.insert(
        0,
        "threshold",
        threshold
    )

    return final_results


def calculate_fairness():

    base = prepare_data()

    all_results = []

    for threshold in THRESHOLDS:

        print(
            f"\nRunning fairness analysis "
            f"at threshold = {threshold}"
        )

        results = calculate_fairness_for_threshold(
            base,
            threshold
        )

        all_results.append(results)

    final_results = pd.concat(
        all_results,
        ignore_index=True
    )

    # Detailed results at 0.60
    detailed_results = final_results[
        final_results["threshold"] == 0.60
    ].copy()

    detailed_results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Threshold sensitivity summary
    sensitivity = (
        final_results
        .groupby(
            ["threshold", "factor"]
        )
        .agg(
            groups=("group", "count"),
            total_candidates=(
                "total_candidates",
                "sum"
            ),
            selected_candidates=(
                "selected_candidates",
                "sum"
            ),
            minimum_selection_rate=(
                "selection_rate",
                "min"
            ),
            maximum_selection_rate=(
                "selection_rate",
                "max"
            ),
            minimum_disparate_impact=(
                "disparate_impact_ratio",
                "min"
            ),
            maximum_disparate_impact=(
                "disparate_impact_ratio",
                "max"
            )
        )
        .reset_index()
    )

    sensitivity.to_csv(
        SENSITIVITY_FILE,
        index=False
    )

    print("\nFairness Evaluation at threshold 0.60:")
    print(
        detailed_results.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        f"Threshold sensitivity saved to: "
        f"{SENSITIVITY_FILE}"
    )


if __name__ == "__main__":
    calculate_fairness()