import pandas as pd
import re


SCREENING_FILE = "screening_output_final/screening_results.csv"
COUNTERFACTUAL_FILE = "final/counterfactual_resumes_valid.csv"
JOB_FILE = "job_requirements_complete.csv"
OUTPUT_FILE = "outputs/context_results.csv"


def extract_years(text):

    text = str(text).lower()

    # Handle numeric month values
    try:
        value = float(text)

        # Experience values in the dataset are months
        if value >= 0:
            return value / 12
    except ValueError:
        pass

    # Handle "X years"
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?",
        text
    )

    if match:
        return float(match.group(1))

    # Handle "X months"
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*months?",
        text
    )

    if match:
        return float(match.group(1)) / 12

    return None


def classify_experience(original, variant, requirement):

    original_years = extract_years(original)
    variant_years = extract_years(variant)
    required_years = extract_years(requirement)

    if None in (
        original_years,
        variant_years,
        required_years
    ):
        return "Needs Review"

    if (
        original_years < required_years
        and variant_years >= required_years
    ):
        return "Required"

    if (
        original_years >= required_years
        and variant_years > original_years
    ):
        return "Potentially Beneficial"

    if variant_years <= original_years:
        return "Not Supported"

    return "Needs Review"


def classify_qualification(
    original,
    variant,
    requirement
):

    requirement = str(requirement).lower()
    original = str(original).lower()
    variant = str(variant).lower()

    original_match = original in requirement
    variant_match = variant in requirement

    if original_match and variant_match:
        return "Accepted"

    if original_match and not variant_match:
        return "Requirement Not Met"

    if variant_match and not original_match:
        return "Required"

    if not original_match and not variant_match:
        return "Not Supported"

    return "Needs Review"


def classify_general(
    original,
    variant,
    requirement
):

    requirement = str(requirement).lower()
    original = str(original).lower()
    variant = str(variant).lower()

    if (
        original not in requirement
        and variant not in requirement
    ):
        return "Not Supported"

    if variant in requirement:
        return "Required"

    return "Needs Review"


def find_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


def analyze_context():

    print("Loading files...")

    screening = pd.read_csv(
        SCREENING_FILE
    )

    counterfactual = pd.read_csv(
        COUNTERFACTUAL_FILE
    )

    jobs = pd.read_csv(
        JOB_FILE
    )

    print(
        f"Screening rows: {len(screening)}"
    )

    print(
        f"Counterfactual rows: {len(counterfactual)}"
    )

    print(
        f"Job rows: {len(jobs)}"
    )

    # --------------------------------------------------
    # Identify useful columns
    # --------------------------------------------------

    cf_resume_col = find_column(
        counterfactual,
        ["resume_id", "id"]
    )

    cf_variant_col = "cf_id"

    job_id_col = find_column(
        jobs,
        ["job_id", "id"]
    )

    # --------------------------------------------------
    # Merge screening with counterfactual information
    # --------------------------------------------------

    if cf_variant_col is None:

        raise ValueError(
            "Could not find variant ID column "
            "in counterfactual dataset."
        )

    merged = screening.merge(
        counterfactual,
        left_on="variant_id",
        right_on=cf_variant_col,
        how="left",
        suffixes=("", "_cf")
    )

    # --------------------------------------------------
    # Merge job requirements
    # --------------------------------------------------

    if job_id_col is not None:

        merged = merged.merge(
            jobs,
            left_on="job_id",
            right_on=job_id_col,
            how="left",
            suffixes=("", "_job")
        )

    results = []

    for _, row in merged.iterrows():

        proxy_type = str(
            row["proxy_type"]
        ).lower()

        # Find original and counterfactual values
        original_value = row.get(
            "original_value",
            ""
        )

        variant_value = row.get(
            "counterfactual_value",
            row.get("proxy_value", "")
        )

        # --------------------------------------------------
        # Determine JD requirement
        # --------------------------------------------------

        requirement = ""

        requirement_columns = [
            "job_description",
            "requirements",
            "required_course",
            "required_degree",
            "minimum_experience",
            "preferred_course",
            "preferred_degree"
        ]

        for column in requirement_columns:

            if column in merged.columns:

                value = row[column]

                if pd.notna(value):

                    requirement += (
                        " " + str(value)
                    )

                # --------------------------------------------------
        # Get structured JD requirements
        # --------------------------------------------------

        accepted_degrees = str(
            row.get("accepted_degrees", "")
        ).split("|")

        accepted_courses = str(
            row.get("accepted_courses", "")
        ).split("|")

        preferred_degree = str(
            row.get("preferred_degree", "")
        )

        preferred_course = str(
            row.get("preferred_course", "")
        )

        minimum_experience = row.get(
            "minimum_total_experience_years"
        )

        jd_requirement = (
            f"Preferred degree: {preferred_degree}; "
            f"Accepted degrees: {', '.join(accepted_degrees)}; "
            f"Preferred course: {preferred_course}; "
            f"Accepted courses: {', '.join(accepted_courses)}; "
            f"Minimum experience: {minimum_experience} years"
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        if proxy_type == "experience":

            try:
                original_years = float(
                    original_value
                ) / 12

                variant_years = float(
                    variant_value
                ) / 12

                required_years = float(
                    minimum_experience
                )

                if (
                    original_years < required_years
                    and variant_years >= required_years
                ):
                    classification = "Required"

                elif (
                    original_years >= required_years
                    and variant_years > original_years
                ):
                    classification = "Potentially Beneficial"

                elif variant_years <= original_years:
                    classification = "Not Supported"

                else:
                    classification = "Needs Review"

            except (ValueError, TypeError):

                classification = "Needs Review"

        elif proxy_type == "degree":

            original_match = any(
                str(original_value).strip().lower()
                == str(x).strip().lower()
                for x in accepted_degrees
            )

            variant_match = any(
                str(variant_value).strip().lower()
                == str(x).strip().lower()
                for x in accepted_degrees
            )

            if original_match and variant_match:
                classification = "Accepted"

            elif original_match and not variant_match:
                classification = "Requirement Not Met"

            elif variant_match and not original_match:
                classification = "Required"

            else:
                classification = "Not Supported"

        elif proxy_type == "course":

            original_match = any(
                str(original_value).strip().lower()
                == str(x).strip().lower()
                for x in accepted_courses
            )

            variant_match = any(
                str(variant_value).strip().lower()
                == str(x).strip().lower()
                for x in accepted_courses
            )

            if original_match and variant_match:
                classification = "Accepted"

            elif original_match and not variant_match:
                classification = "Requirement Not Met"

            elif variant_match and not original_match:
                classification = "Required"

            else:
                classification = "Not Supported"

        else:

            # Name, college, city, age, etc.
            # are not explicitly required by the current JD schema.

            classification = "Not Supported"

        # --------------------------------------------------
        # Score information
        # --------------------------------------------------

        base_score = float(
            row["base_score"]
        )

        variant_score = float(
            row["variant_score"]
        )

        delta = (
            variant_score - base_score
        )

        results.append({

            "resume_id":
                row["original_resume_id"],

            "job_id":
                row["job_id"],

            "variant_id":
                row["variant_id"],

            "proxy_type":
                proxy_type,

            "original_value":
                original_value,

            "variant_value":
                variant_value,

            "jd_requirement":
                jd_requirement,

            "base_score":
                base_score,

            "variant_score":
                variant_score,

            "delta":
                delta,

            "context_classification":
                classification
        })

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nContext Analysis Results:"
    )

    print(
        results_df.head(20)
        .to_string(index=False)
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        "\nClassification counts:"
    )

    print(
        results_df[
            "context_classification"
        ].value_counts()
    )


if __name__ == "__main__":

    analyze_context()

