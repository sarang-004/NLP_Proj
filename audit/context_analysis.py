import pandas as pd
import re


def extract_years(text):
    """
    Extract the first numerical experience value.

    Examples:
        "2 years" -> 2
        "5 years experience" -> 5
        "2+ years experience" -> 2
    """

    text = str(text).lower()

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?",
        text
    )

    if match:
        return float(match.group(1))

    return None


def classify_experience(
    original_value,
    variant_value,
    jd_requirement
):
    """
    Classify whether an experience difference
    is supported by the job context.
    """

    original_years = extract_years(original_value)
    variant_years = extract_years(variant_value)
    required_years = extract_years(jd_requirement)

    if (
        original_years is None
        or variant_years is None
        or required_years is None
    ):
        return "Needs Review"

    # Variant meets the requirement,
    # while original does not.
    if (
        original_years < required_years
        and variant_years >= required_years
    ):
        return "Required"

    # Both meet the minimum requirement,
    # but the variant has more experience.
    if (
        original_years >= required_years
        and variant_years > original_years
    ):
        return "Potentially Beneficial"

    # Variant does not improve qualification
    # relative to the original.
    if variant_years <= original_years:
        return "Not Supported"

    return "Needs Review"


def classify_course(
    original_value,
    variant_value,
    jd_requirement
):
    """
    Classify course/degree differences
    using the job requirement.
    """

    requirement = str(jd_requirement).lower()

    original = str(original_value).lower()
    variant = str(variant_value).lower()

    original_accepted = original in requirement
    variant_accepted = variant in requirement

    # Both qualifications are explicitly accepted
    if original_accepted and variant_accepted:
        return "Accepted"

    # Original satisfies the requirement,
    # but the variant does not.
    if original_accepted and not variant_accepted:
        return "Requirement Not Met"

    # Variant satisfies the requirement,
    # but original does not.
    if variant_accepted and not original_accepted:
        return "Required"

    # Neither qualification is mentioned.
    if not original_accepted and not variant_accepted:
        return "Not Supported"

    return "Needs Review"


def classify_general_context(
    original_value,
    variant_value,
    jd_requirement
):
    """
    Handle contextual factors such as
    college and city.
    """

    requirement = str(jd_requirement).lower()

    original = str(original_value).lower()
    variant = str(variant_value).lower()

    # If neither value appears in the JD,
    # there is no explicit job requirement.
    if (
        original not in requirement
        and variant not in requirement
    ):
        return "Not Supported"

    # If the variant is explicitly mentioned
    if variant in requirement:
        return "Required"

    return "Needs Review"


def analyze_context(input_file, output_file):

    df = pd.read_csv(input_file)

    results = []

    for _, row in df.iterrows():

        proxy_type = str(
            row["proxy_type"]
        ).lower()

        original_value = str(
            row["original_value"]
        )

        variant_value = str(
            row["variant_value"]
        )

        jd_requirement = str(
            row["jd_requirement"]
        )

        base_score = float(
            row["base_score"]
        )

        variant_score = float(
            row["variant_score"]
        )

        delta = variant_score - base_score

        # Choose classification method
        if proxy_type == "experience":

            classification = classify_experience(
                original_value,
                variant_value,
                jd_requirement
            )

        elif proxy_type in ["course", "degree"]:

            classification = classify_course(
                original_value,
                variant_value,
                jd_requirement
            )

        else:

            classification = classify_general_context(
                original_value,
                variant_value,
                jd_requirement
            )

        results.append({
            "resume_id": row["resume_id"],
            "proxy_type": proxy_type,
            "original_value": original_value,
            "variant_value": variant_value,
            "jd_requirement": jd_requirement,
            "base_score": base_score,
            "variant_score": variant_score,
            "delta": delta,
            "context_classification": classification
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nContext Analysis Results:")
    print(
        results_df.to_string(index=False)
    )


if __name__ == "__main__":

    analyze_context(
        "data/mock_context.csv",
        "outputs/context_results.csv"
    )