import pandas as pd
import re


# ==========================================================
# FILE PATHS
# ==========================================================

SCREENING_FILE = "screening_output_final/screening_results.csv"

COUNTERFACTUAL_FILE = (
    "final/counterfactual_resumes_valid.csv"
)

JOB_FILE = "job_requirements_complete.csv"

COMPANY_COURSE_FILE = (
    "final/company_course_preferences.csv"
)

OUTPUT_FILE = "outputs/context_results.csv"


# ==========================================================
# TEXT NORMALIZATION
# ==========================================================

def normalize(value):

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


# ==========================================================
# EXPERIENCE UTILITIES
# ==========================================================

def extract_years(text):

    text = str(text).lower()

    try:

        value = float(text)

        if value >= 0:
            return value / 12

    except ValueError:
        pass

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?",
        text
    )

    if match:
        return float(match.group(1))

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*months?",
        text
    )

    if match:
        return float(match.group(1)) / 12

    return None


# ==========================================================
# COURSE CLASSIFICATION
# ==========================================================

def classify_course(
    course,
    preferred_course,
    accepted_courses
):

    course = normalize(course)

    preferred_course = normalize(
        preferred_course
    )

    accepted_courses = [
        normalize(x)
        for x in str(
            accepted_courses
        ).split("|")
    ]

    if course == "":
        return "Unknown"

    if course == preferred_course:
        return "Preferred"

    if course in accepted_courses:
        return "Accepted"

    return "Unrelated"


# ==========================================================
# COURSE TRANSFORMATION INTERPRETATION
# ==========================================================

def interpret_course_change(
    original_status,
    variant_status
):

    if (
        original_status == "Unknown"
        or variant_status == "Unknown"
    ):
        return "Needs Review"

    if (
        original_status == "Preferred"
        and variant_status == "Preferred"
    ):
        return "No Preference Change"

    if (
        original_status == "Preferred"
        and variant_status == "Accepted"
    ):
        return "Potentially Less Preferred"

    if (
        original_status == "Preferred"
        and variant_status == "Unrelated"
    ):
        return "Requirement Not Met"

    if (
        original_status == "Accepted"
        and variant_status == "Preferred"
    ):
        return "Potentially Beneficial"

    if (
        original_status == "Accepted"
        and variant_status == "Accepted"
    ):
        return "Equivalent Acceptance"

    if (
        original_status == "Accepted"
        and variant_status == "Unrelated"
    ):
        return "Requirement Not Met"

    if (
        original_status == "Unrelated"
        and variant_status == "Preferred"
    ):
        return "Strongly Beneficial"

    if (
        original_status == "Unrelated"
        and variant_status == "Accepted"
    ):
        return "Potentially Beneficial"

    if (
        original_status == "Unrelated"
        and variant_status == "Unrelated"
    ):
        return "No Relevant Improvement"

    return "Needs Review"


# ==========================================================
# DEGREE CLASSIFICATION
#
# Kept separate from course classification.
# This allows B.Tech, M.Tech and Integrated M.Tech
# to be evaluated independently from the course.
# ==========================================================

def classify_degree_change(
    original,
    variant,
    accepted_degrees
):

    original = normalize(original)

    variant = normalize(variant)

    accepted_degrees = [
        normalize(x)
        for x in str(
            accepted_degrees
        ).split("|")
    ]

    original_match = (
        original in accepted_degrees
    )

    variant_match = (
        variant in accepted_degrees
    )

    if original_match and variant_match:
        return "Accepted"

    if original_match and not variant_match:
        return "Requirement Not Met"

    if variant_match and not original_match:
        return "Potentially Beneficial"

    return "Not Supported"


# ==========================================================
# EXPERIENCE CLASSIFICATION
# ==========================================================

def classify_experience_change(
    original,
    variant,
    minimum_experience
):

    try:

        original_years = (
            float(original) / 12
        )

        variant_years = (
            float(variant) / 12
        )

        required_years = float(
            minimum_experience
        )

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

    except (
        ValueError,
        TypeError
    ):

        return "Needs Review"


# ==========================================================
# MAIN CONTEXT ANALYSIS
# ==========================================================

def analyze_context():

    print(
        "\nLoading files..."
    )

    screening = pd.read_csv(
        SCREENING_FILE
    )

    counterfactual = pd.read_csv(
        COUNTERFACTUAL_FILE
    )

    jobs = pd.read_csv(
        JOB_FILE
    )

    company_courses = pd.read_csv(
        COMPANY_COURSE_FILE
    )

    print(
        f"Screening rows: "
        f"{len(screening)}"
    )

    print(
        f"Counterfactual rows: "
        f"{len(counterfactual)}"
    )

    print(
        f"Job rows: "
        f"{len(jobs)}"
    )

    print(
        f"Company-course mappings: "
        f"{len(company_courses)}"
    )


    # ======================================================
    # VALIDATE COMPANY-COURSE MAPPING
    # ======================================================

    mapping_job_ids = set(
        company_courses["job_id"]
    )

    job_ids = set(
        jobs["job_id"]
    )

    missing_mappings = (
        job_ids - mapping_job_ids
    )

    if missing_mappings:

        raise ValueError(
            "Missing company-course mappings "
            f"for jobs: {missing_mappings}"
        )


    # ======================================================
    # MERGE SCREENING + COUNTERFACTUAL DATA
    # ======================================================

    merged = screening.merge(
        counterfactual,
        left_on="variant_id",
        right_on="cf_id",
        how="left",
        suffixes=("", "_cf")
    )


    # ======================================================
    # MERGE JOB DATA
    # ======================================================

    merged = merged.merge(
        jobs,
        on="job_id",
        how="left",
        suffixes=("", "_job")
    )


    # ======================================================
    # MERGE COMPANY-COURSE CONTEXT
    # ======================================================

    company_context = company_courses[
        [
            "company",
            "job_id",
            "preferred_course",
            "accepted_courses"
        ]
    ].copy()

    # Rename to prevent ambiguity with JD columns
    company_context = company_context.rename(
        columns={
            "preferred_course":
                "company_preferred_course",

            "accepted_courses":
                "company_accepted_courses"
        }
    )

    merged = merged.merge(
        company_context,
        on="job_id",
        how="left"
    )


    # ======================================================
    # CHECK MERGE QUALITY
    # ======================================================

    missing_company = merged[
        "company"
    ].isna().sum()

    if missing_company > 0:

        print(
            "\nWARNING:"
        )

        print(
            f"{missing_company} rows "
            "do not have company context."
        )


    results = []


    # ======================================================
    # PROCESS EACH COUNTERFACTUAL
    # ======================================================

    for _, row in merged.iterrows():

        proxy_type = normalize(
            row.get(
                "proxy_type",
                ""
            )
        )

        original_value = row.get(
            "original_value",
            ""
        )

        variant_value = row.get(
            "counterfactual_value",
            row.get(
                "proxy_value",
                ""
            )
        )


        # --------------------------------------------------
        # Get company context
        # --------------------------------------------------

        company = row.get(
            "company",
            ""
        )

        company_preferred_course = row.get(
            "company_preferred_course",
            ""
        )

        company_accepted_courses = row.get(
            "company_accepted_courses",
            ""
        )


        # --------------------------------------------------
        # Build readable JD requirement
        # --------------------------------------------------

        accepted_degrees = row.get(
            "accepted_degrees",
            ""
        )

        preferred_degree = row.get(
            "preferred_degree",
            ""
        )

        minimum_experience = row.get(
            "minimum_total_experience_years",
            None
        )

        jd_requirement = (
            f"Company: {company}; "
            f"Preferred course: "
            f"{company_preferred_course}; "
            f"Accepted courses: "
            f"{company_accepted_courses}; "
            f"Preferred degree: "
            f"{preferred_degree}; "
            f"Accepted degrees: "
            f"{accepted_degrees}; "
            f"Minimum experience: "
            f"{minimum_experience} years"
        )


        # --------------------------------------------------
        # Default course fields
        # --------------------------------------------------

        original_course_status = ""

        variant_course_status = ""


        # ==================================================
        # EXPERIENCE
        # ==================================================

        if proxy_type == "experience":

            classification = (
                classify_experience_change(
                    original_value,
                    variant_value,
                    minimum_experience
                )
            )


        # ==================================================
        # DEGREE
        #
        # Degree level is evaluated separately.
        # ==================================================

        elif proxy_type == "degree":

            classification = (
                classify_degree_change(
                    original_value,
                    variant_value,
                    accepted_degrees
                )
            )


        # ==================================================
        # COURSE
        #
        # Uses company-course preference mapping.
        # ==================================================

        elif proxy_type == "course":

            original_course_status = (
                classify_course(
                    original_value,
                    company_preferred_course,
                    company_accepted_courses
                )
            )

            variant_course_status = (
                classify_course(
                    variant_value,
                    company_preferred_course,
                    company_accepted_courses
                )
            )

            classification = (
                interpret_course_change(
                    original_course_status,
                    variant_course_status
                )
            )


        # ==================================================
        # OTHER PROXIES
        # ==================================================

        else:

            classification = (
                "Not Supported"
            )


        # ==================================================
        # SCORE INFORMATION
        # ==================================================

        try:

            base_score = float(
                row["base_score"]
            )

            variant_score = float(
                row["variant_score"]
            )

            delta = (
                variant_score
                - base_score
            )

        except (
            ValueError,
            TypeError
        ):

            base_score = None

            variant_score = None

            delta = None


        # ==================================================
        # STORE RESULT
        # ==================================================

        results.append({

            "resume_id":
                row.get(
                    "original_resume_id",
                    row.get(
                        "resume_id",
                        ""
                    )
                ),

            "job_id":
                row.get(
                    "job_id",
                    ""
                ),

            "company":
                company,

            "variant_id":
                row.get(
                    "variant_id",
                    ""
                ),

            "proxy_type":
                proxy_type,

            "original_value":
                original_value,

            "variant_value":
                variant_value,

            "original_course_status":
                original_course_status,

            "variant_course_status":
                variant_course_status,

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


    # ======================================================
    # CREATE OUTPUT
    # ======================================================

    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ======================================================
    # DISPLAY RESULTS
    # ======================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "CONTEXT ANALYSIS RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTotal rows: "
        f"{len(results_df)}"
    )

    print(
        "\nFirst 20 results:\n"
    )

    print(
        results_df.head(20)
        .to_string(index=False)
    )


    print(
        "\nClassification counts:"
    )

    print(
        results_df[
            "context_classification"
        ]
        .value_counts()
    )


    # ======================================================
    # COURSE-SPECIFIC SUMMARY
    # ======================================================

    course_results = results_df[
        results_df["proxy_type"] == "course"
    ]


    if len(course_results) > 0:

        print(
            "\n" + "=" * 80
        )

        print(
            "COMPANY-COURSE CONTEXT SUMMARY"
        )

        print(
            "=" * 80
        )

        print(
            f"\nCourse transformations: "
            f"{len(course_results)}"
        )

        print(
            "\nOriginal course status:"
        )

        print(
            course_results[
                "original_course_status"
            ]
            .value_counts()
        )

        print(
            "\nVariant course status:"
        )

        print(
            course_results[
                "variant_course_status"
            ]
            .value_counts()
        )

        print(
            "\nCourse interpretation:"
        )

        print(
            course_results[
                "context_classification"
            ]
            .value_counts()
        )


    print(
        "\n" + "=" * 80
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print(
        "=" * 80
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    analyze_context()
