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
    except (ValueError, TypeError):
        return "Needs Review"


def _context_missing(value):
    return normalize(value) in {"", "na", "n/a", "nan", "none", "null", "unknown"}


def _accepted_values(value):
    return {normalize(item) for item in str(value or "").split("|") if normalize(item)}


def classify_contextual_case(
    proxy_type,
    original_value,
    variant_value,
    base_score,
    variant_score,
    job,
    component_changes=None,
):
    """Classify a measured transformation with a rationale and evidence.

    This is an audit interpretation only. It never changes measured scores.
    An unlisted course is unknown unless the JD mapping explicitly lists it as
    unrelated; no unsupported course taxonomy is inferred here.
    """
    attr = normalize(proxy_type)
    job = job if hasattr(job, "get") else {}
    components = component_changes or {}
    evidence = []
    try:
        delta = float(variant_score) - float(base_score)
    except (TypeError, ValueError):
        delta = None
    if delta is not None:
        evidence.append(f"Measured final-score change: {delta:+.6f}")
    for name, value in components.items():
        try:
            component_delta = float(value)
        except (TypeError, ValueError):
            continue
        if abs(component_delta) > 1e-9:
            evidence.append(f"Measured {name} component change: {component_delta:+.6f}")

    if "+" in attr:
        return {
            "classification": "Requires review",
            "rationale": "This case changes multiple attributes, so the effect of each attribute cannot be isolated by this classification.",
            "job_requirement": "Joint transformations require compatible individual-factor evidence.",
            "supporting_evidence": evidence or ["Joint attribute values are recorded, but individual attribution is unavailable."],
        }
    if _context_missing(original_value) or _context_missing(variant_value) or delta is None:
        return {
            "classification": "Requires review",
            "rationale": "The original value, transformed value, or measured score is unavailable.",
            "job_requirement": "Unavailable",
            "supporting_evidence": evidence + ["Missing transformation or score evidence."],
        }

    if attr in {"name", "age", "college", "city"}:
        relevant_text = " ".join(str(job.get(key, "")) for key in ("job_description", "job_title", "required_skills")).lower()
        explicitly_required = False
        if attr == "city":
            explicitly_required = any(term in relevant_text for term in ("must be located", "based in", "onsite in", "on-site in", "relocation required"))
        if attr == "college":
            explicitly_required = any(term in relevant_text for term in ("accredited institution", "approved institution", "college requirement"))
        if explicitly_required:
            return {
                "classification": "Requires review",
                "rationale": "The JD appears to mention a location or institution condition; verify that it is necessary and applied consistently.",
                "job_requirement": "Explicit JD text requires review: " + str(job.get("job_description", "")),
                "supporting_evidence": evidence + ["A matching job-context phrase was found; this is not by itself proof of a legitimate criterion."],
            }
        if abs(delta) > 1e-9:
            return {
                "classification": "Potentially biased",
                "rationale": f"The measured score changed when {attr} changed, and the available job requirements do not establish {attr} as a hiring criterion.",
                "job_requirement": "No job-relevant requirement supporting this attribute was found in the supplied JD fields.",
                "supporting_evidence": evidence + [f"Changed attribute: {attr}; original={original_value}; variant={variant_value}. This is model sensitivity, not proof of discrimination."],
            }
        return {
            "classification": "Unsupported",
            "rationale": f"No measured final-score change was observed for {attr}; the JD provides no job-relevant basis for using it as a criterion.",
            "job_requirement": "No relevant criterion specified.",
            "supporting_evidence": evidence + [f"Changed attribute: {attr}; original={original_value}; variant={variant_value}."],
        }

    if attr in {"degree", "course"}:
        if attr == "degree":
            preferred = normalize(job.get("preferred_degree", ""))
            accepted = _accepted_values(job.get("accepted_degrees", ""))
            requirement = f"Preferred degree: {preferred or 'not specified'}; accepted degrees: {'|'.join(sorted(accepted)) or 'not specified'}"
        else:
            preferred = normalize(job.get("preferred_course", ""))
            accepted = _accepted_values(job.get("accepted_courses", ""))
            unrelated = _accepted_values(job.get("unrelated_courses", ""))
            requirement = f"Preferred course: {preferred or 'not specified'}; accepted courses: {'|'.join(sorted(accepted)) or 'not specified'}"
        if not preferred and not accepted:
            return {"classification": "Requires review", "rationale": f"The JD has no usable {attr} requirement or preference mapping.", "job_requirement": requirement, "supporting_evidence": evidence + ["Qualification context unavailable."]}
        def status(value):
            value = normalize(value)
            if _context_missing(value):
                return "unavailable"
            if preferred and value == preferred:
                return "preferred"
            if value in accepted:
                return "accepted"
            if attr == "course" and value not in unrelated:
                return "unclassified"
            return "not listed"
        before, after = status(original_value), status(variant_value)
        evidence.append(f"{attr.title()} standing under listed requirements: {before} → {after}")
        if "unavailable" in (before, after) or "unclassified" in (before, after):
            return {"classification": "Requires review", "rationale": f"The {attr} value cannot be compared reliably with the supplied requirement mapping.", "job_requirement": requirement, "supporting_evidence": evidence}
        qualification_delta = components.get("education_course", 0.0)
        try:
            qualification_delta = float(qualification_delta)
        except (TypeError, ValueError):
            qualification_delta = 0.0
        if before != after and abs(qualification_delta) > 1e-9:
            return {"classification": "Job-relevant", "rationale": f"The changed {attr} moves between distinct qualification statuses listed in the job requirements, and the combined education/course component changed.", "job_requirement": requirement, "supporting_evidence": evidence}
        if abs(delta) > 1e-9 and abs(qualification_delta) <= 1e-9:
            return {"classification": "Potentially biased", "rationale": f"The total score changed without a measured change in the combined education/course score; available evidence does not support attributing the change to the listed {attr} requirement.", "job_requirement": requirement, "supporting_evidence": evidence}
        return {"classification": "Unsupported", "rationale": f"The available score and qualification evidence does not show a job-relevant effect from this {attr} transformation.", "job_requirement": requirement, "supporting_evidence": evidence}

    if attr in {"experience", "total_experience_years"}:
        minimum = job.get("minimum_total_experience_years", "")
        try:
            required_years = float(minimum)
            def years(value):
                text = str(value).strip().lower()
                match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(years?|months?)?", text)
                if not match:
                    return None
                amount = float(match.group(1))
                if match.group(2) and match.group(2).startswith("year"):
                    return amount
                if match.group(2) and match.group(2).startswith("month"):
                    return amount / 12
                # Existing counterfactual source convention stores experience
                # in months; explicit unitless values follow that schema.
                return amount if attr == "total_experience_years" else amount / 12
            before, after = years(original_value), years(variant_value)
        except (TypeError, ValueError):
            required_years, before, after = None, None, None
        if required_years is None or before is None or after is None:
            return {"classification": "Requires review", "rationale": "Experience values or the JD minimum experience requirement are unavailable or ambiguous.", "job_requirement": f"Minimum experience: {minimum or 'not specified'} years", "supporting_evidence": evidence}
        evidence.append(f"Explicit experience requirement: {required_years:g} years; original={before:g} years; variant={after:g} years")
        return {"classification": "Job-relevant", "rationale": "The JD specifies a minimum experience requirement and both transformed values can be compared against it.", "job_requirement": f"Minimum experience: {required_years:g} years", "supporting_evidence": evidence}

    if attr == "employment_gap_months":
        if abs(delta) > 1e-9:
            return {"classification": "Potentially biased", "rationale": "The score changed when the employment gap changed, but the existing JD schema supplies no job requirement that makes gap duration relevant.", "job_requirement": "No employment-gap criterion is recorded in the job schema.", "supporting_evidence": evidence + ["The existing model has no explicit employment-gap component; inspect semantic component changes for the measured path."]}
        return {"classification": "Unsupported", "rationale": "No measured score change was observed and no employment-gap criterion is recorded in the JD.", "job_requirement": "No employment-gap criterion is recorded.", "supporting_evidence": evidence}

    return {"classification": "Requires review", "rationale": "The changed attribute is not covered by the contextual rules.", "job_requirement": "Unavailable", "supporting_evidence": evidence}

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

        context_result = classify_contextual_case(
            proxy_type=proxy_type,
            original_value=original_value,
            variant_value=variant_value,
            base_score=base_score,
            variant_score=variant_score,
            job=row,
            component_changes={
                "skills": row.get("skills_delta", 0),
                "experience": row.get("experience_delta", 0),
                "education_course": row.get("education_course_delta", 0),
                "projects": row.get("projects_delta", 0),
                "semantic": row.get("semantic_delta", 0),
            },
        )
        classification = context_result["classification"]


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

            "base_score_methodology_version": row.get("base_score_methodology_version", ""),

            "variant_score_methodology_version": row.get("variant_score_methodology_version", ""),

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
                classification,

            "context_rationale":
                context_result["rationale"],

            "contextual_job_requirement":
                context_result["job_requirement"],

            "supporting_evidence":
                " | ".join(context_result["supporting_evidence"]),
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
