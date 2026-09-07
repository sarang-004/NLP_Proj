from audit.context_analysis import (
    classify_course,
    interpret_course_change,
    classify_degree_change,
)


# ==========================================================
# COMPANY-COURSE PREFERENCE TESTS
# ==========================================================

def test_preferred_to_accepted():
    """
    Preferred course changed to an accepted but non-preferred course.
    """

    preferred = "Computer Science and Engineering"
    accepted = (
        "Computer Science and Engineering|"
        "Information Technology|"
        "Computer Applications"
    )

    original = classify_course(
        "Computer Science and Engineering",
        preferred,
        accepted
    )

    variant = classify_course(
        "Information Technology",
        preferred,
        accepted
    )

    assert original == "Preferred"
    assert variant == "Accepted"

    assert interpret_course_change(
        original,
        variant
    ) == "Potentially Less Preferred"


def test_accepted_to_preferred():
    """
    Accepted course changed to the company's preferred course.
    """

    preferred = "Computer Science and Engineering"

    accepted = (
        "Computer Science and Engineering|"
        "Information Technology|"
        "Computer Applications"
    )

    original = classify_course(
        "Information Technology",
        preferred,
        accepted
    )

    variant = classify_course(
        "Computer Science and Engineering",
        preferred,
        accepted
    )

    assert original == "Accepted"
    assert variant == "Preferred"

    assert interpret_course_change(
        original,
        variant
    ) == "Potentially Beneficial"


def test_accepted_to_unrelated():
    """
    Accepted course changed to an unrelated course.
    """

    preferred = "Computer Science and Engineering"

    accepted = (
        "Computer Science and Engineering|"
        "Information Technology|"
        "Computer Applications"
    )

    original = classify_course(
        "Information Technology",
        preferred,
        accepted
    )

    variant = classify_course(
        "Civil Engineering",
        preferred,
        accepted
    )

    assert original == "Accepted"
    assert variant == "Unrelated"

    assert interpret_course_change(
        original,
        variant
    ) == "Requirement Not Met"


def test_unrelated_to_accepted():
    """
    Unrelated course changed to an accepted course.
    """

    preferred = "Computer Science and Engineering"

    accepted = (
        "Computer Science and Engineering|"
        "Information Technology|"
        "Computer Applications"
    )

    original = classify_course(
        "Civil Engineering",
        preferred,
        accepted
    )

    variant = classify_course(
        "Information Technology",
        preferred,
        accepted
    )

    assert original == "Unrelated"
    assert variant == "Accepted"

    assert interpret_course_change(
        original,
        variant
    ) == "Potentially Beneficial"


def test_unrelated_to_preferred():
    """
    Unrelated course changed directly to the preferred course.
    """

    preferred = "Computer Science and Engineering"

    accepted = (
        "Computer Science and Engineering|"
        "Information Technology|"
        "Computer Applications"
    )

    original = classify_course(
        "Civil Engineering",
        preferred,
        accepted
    )

    variant = classify_course(
        "Computer Science and Engineering",
        preferred,
        accepted
    )

    assert original == "Unrelated"
    assert variant == "Preferred"

    assert interpret_course_change(
        original,
        variant
    ) == "Strongly Beneficial"


def test_preferred_to_unrelated():
    """
    Preferred course changed to an unrelated course.
    """

    preferred = "Civil Engineering"

    accepted = (
        "Civil Engineering|"
        "Construction Engineering|"
        "Structural Engineering"
    )

    original = classify_course(
        "Civil Engineering",
        preferred,
        accepted
    )

    variant = classify_course(
        "Mechanical Engineering",
        preferred,
        accepted
    )

    assert original == "Preferred"
    assert variant == "Unrelated"

    assert interpret_course_change(
        original,
        variant
    ) == "Requirement Not Met"


def test_same_preferred_course():
    """
    Preferred course remains unchanged.
    """

    preferred = "Data Science"

    accepted = (
        "Data Science|"
        "Computer Science|"
        "Artificial Intelligence|"
        "Statistics"
    )

    original = classify_course(
        "Data Science",
        preferred,
        accepted
    )

    variant = classify_course(
        "Data Science",
        preferred,
        accepted
    )

    assert original == "Preferred"
    assert variant == "Preferred"

    assert interpret_course_change(
        original,
        variant
    ) == "No Preference Change"


# ==========================================================
# DEGREE TESTS
# ==========================================================

def test_degree_accepted_to_accepted():
    """
    Both degrees are accepted by the JD.
    """

    accepted_degrees = "B.Tech|B.E.|M.Tech|MCA"

    result = classify_degree_change(
        "B.Tech",
        "M.Tech",
        accepted_degrees
    )

    assert result == "Accepted"


def test_degree_accepted_to_unaccepted():
    """
    Accepted degree changed to an unsupported degree.
    """

    accepted_degrees = "B.Tech|B.E.|M.Tech|MCA"

    result = classify_degree_change(
        "B.Tech",
        "Diploma",
        accepted_degrees
    )

    assert result == "Requirement Not Met"


def test_degree_unaccepted_to_accepted():
    """
    Unsupported degree changed to an accepted degree.
    """

    accepted_degrees = "B.Tech|B.E.|M.Tech|MCA"

    result = classify_degree_change(
        "Diploma",
        "M.Tech",
        accepted_degrees
    )

    assert result == "Potentially Beneficial"


def test_degree_btech_to_mtech_integrated():
    """
    Example required by the project:
    B.Tech Core -> Integrated M.Tech.

    This test intentionally checks degree support,
    not company-course preference.
    """

    accepted_degrees = (
        "B.Tech|"
        "M.Tech|"
        "Integrated M.Tech|"
        "B.E."
    )

    result = classify_degree_change(
        "B.Tech",
        "Integrated M.Tech",
        accepted_degrees
    )

    assert result == "Accepted"



# ==========================================================
# COMPANY-COURSE MAPPING DATA VALIDATION
# ==========================================================

def test_company_course_mapping_integrity():
    """
    Validate the complete company-course preference mapping.
    """

    import pandas as pd

    mapping_file = "final/company_course_preferences.csv"
    job_file = "job_requirements_complete.csv"

    mappings = pd.read_csv(mapping_file)
    jobs = pd.read_csv(job_file)

    # Every job should have exactly one mapping
    assert len(mappings) == len(jobs)
    assert mappings["job_id"].nunique() == len(jobs)

    # Every JD job_id must exist in the mapping
    assert set(jobs["job_id"]) == set(mappings["job_id"])

    # Required company-course fields must not be empty
    assert mappings["company"].notna().all()
    assert mappings["preferred_course"].notna().all()
    assert mappings["accepted_courses"].notna().all()

    # Preferred course must be included among accepted courses
    for _, row in mappings.iterrows():

        preferred = str(
            row["preferred_course"]
        ).strip().lower()

        accepted = [
            x.strip().lower()
            for x in str(
                row["accepted_courses"]
            ).split("|")
        ]

        assert preferred in accepted
