import pandas as pd

from data_layer import (
    NA,
    extract_age,
    extract_course,
    extract_degree,
    extract_degrees,
    extract_experience_years,
    extract_experience_records,
    extract_projects,
    extract_skills,
    structure_resume,
    transform_counterfactual,
    validate_data,
)


def test_degree_and_course_are_separate():
    edu = "B.Tech Computer Science and Engineering, IIT Bombay"
    assert extract_degree(edu) == "B.Tech"
    assert extract_course(edu)[0] == "Computer Science and Engineering"


def test_degree_level_not_mistaken_for_course():
    assert extract_degree("Master of Commerce Marketing") == "M.Com"
    assert extract_course("Master of Commerce Marketing Mumbai") == ("Marketing", "Marketing")


def test_highest_degree_selected_and_all_degrees_preserved():
    for text, expected in [
        ("Master of Engineering Information Technology B.E. Information Technology", "M.E."),
        ("M.Tech Computer Engineering B.E. Computer Science", "M.Tech"),
        ("MBA Finance B.Com Computer", "MBA"),
    ]:
        assert extract_degree(text) == expected
        assert len(extract_degrees(text)) == 2


def test_course_is_associated_with_primary_degree_not_diploma():
    education = "Post Graduate Diploma Information Technology Pune Symbiosis Institute B.E. Electronics and Telecommunications Mumbai University"
    assert extract_degree(education) == "B.E."
    assert extract_course(education)[0] == "Electronics and Telecommunications"
    assert extract_course("MBA Finance B.Com Computer")[0] == "Finance"


def test_job_title_does_not_become_course():
    assert extract_course("B.Com Commerce Mumbai University Operations Manager") == ("Commerce", "Commerce")


def test_age_requires_explicit_age_label():
    assert extract_age("Age: 29; graduated in 2019; 5 years experience") == "29"
    assert extract_age("graduated in 2019; 5 years experience") == NA


def test_experience_uses_explicit_employment_duration_only():
    assert extract_experience_years("Worked at A for 3 years.") == "3.0"
    assert extract_experience_years("Worked at A for 3 years. Worked at B for 18 months.") == NA
    assert extract_experience_years("Python - Experience - 72 months") == NA


def test_employment_date_ranges_union_overlaps():
    text = "Worked as analyst from Jan 2018 to Jan 2021. Working as manager from Jan 2020 to Jan 2023."
    assert extract_experience_years(text) == "5.08"


def test_project_date_range_is_not_employment_experience():
    assert extract_experience_years("Project duration: Jan 2018 - Jan 2021. Built software for the client.") == NA


def test_multiple_undated_employment_durations_are_not_summed():
    assert extract_experience_years("Worked at A for 3 years. Worked at B for 2 years.") == NA


def test_numeric_month_year_employment_dates_use_as_of_and_merge_overlaps():
    from datetime import date

    text = "Acme Labs (01.2020 - 01.2022)\nOther Inc (01.2021 - 01.2023)"
    assert extract_experience_years(text, as_of=date(2024, 1, 1)) == "3.08"
    assert extract_experience_years("GogoApps (01.2018 - now)", as_of=date(2024, 1, 1)) == "6.08"


def test_known_midhunlal_resume_roles_and_inclusive_internship_months():
    parsed = extract_experience_records("""EXPERIENCE
K7 Computing Pvt. Ltd., Chennai, India — Cybersecurity Intern | May 2025 – June 2025
Gained hands-on experience with cybersecurity tools.
Star Systems India Pvt. Ltd., India — AI & Machine Learning Intern | May 2026 – July 2026
Completed an internship focused on AI and Machine Learning.
CERTIFICATIONS
AI & Machine Learning Internship — Star Systems India Pvt. Ltd. | May 2026 – July 2026""", "Midhunlal P")
    assert [(r["organization"], r["role_title"], r["experience_category"], r["duration_months"])
            for r in parsed["records"]] == [
        ("K7 Computing Pvt. Ltd.", "Cybersecurity Intern", "internship", 2),
        ("Star Systems India Pvt. Ltd.", "AI & Machine Learning Intern", "internship", 3),
    ]
    assert parsed["combined_months_by_category"]["internship"] == 5
    assert parsed["total_professional_months"] == 5


def test_known_gautham_resume_separates_internship_and_student_leadership():
    parsed = extract_experience_records("""EXPERIENCE
Retailcloud — Software Engineering Intern
May 2026 – July 2026
Designed and built a backend application using Spring Boot.
CodeChef - VIT Student Chapter — Senior Core
March 2024 – February 2026
Developed frontend interfaces and co-organized a technical event.
PROJECTS
Study project — Developer
January 2023 to December 2023""", "Gautham Praveen")
    assert [(r["organization"], r["role_title"], r["experience_category"], r["duration_months"])
            for r in parsed["records"]] == [
        ("Retailcloud", "Software Engineering Intern", "internship", 3),
        ("CodeChef - VIT Student Chapter", "Senior Core", "student_leadership_or_extracurricular", 24),
        ("Study project", "Developer", "academic_or_personal_project", 12),
    ]
    assert parsed["combined_months_by_category"] == {
        "internship": 3,
        "student_leadership_or_extracurricular": 24,
        "academic_or_personal_project": 12,
    }
    assert parsed["total_professional_months"] == 3


def test_date_parser_supports_abbreviations_numeric_dates_unicode_and_wrapped_ranges():
    from datetime import date

    cases = [
        ("Jan 2024 - Dec 2025", 24),
        ("2024-01 to 2025-12", 24),
        ("January 2024\nto\nDecember 2025", 24),
        ("2024 – 2025", 24),
        ("May 2025 — Present", 18),
    ]
    for date_text, expected_months in cases:
        parsed = extract_experience_records(
            f"EXPERIENCE\nAcme Corp — Software Engineer\n{date_text}", "R", date(2026, 10, 15)
        )
        assert parsed["records"][0]["duration_months"] == expected_months
        assert parsed["records"][0]["raw_date_text"]
    ambiguous = extract_experience_records("EXPERIENCE\nAcme Corp — Engineer\n13/2024 - 15/2024", "R")["records"]
    assert len(ambiguous) == 1
    assert ambiguous[0]["duration_months"] is None
    assert ambiguous[0]["raw_date_text"] == "13/2024 - 15/2024"
    assert ambiguous[0]["extraction_confidence"] == "review"
    assert extract_experience_records("EXPERIENCE\nAcme Corp — Engineer\nNo dates provided", "R")["records"] == []


def test_project_ranges_do_not_enter_professional_total_and_overlaps_union_by_category():
    parsed = extract_experience_records("""EXPERIENCE
Acme Corp — Full-time Software Engineer | Jan 2020 - Dec 2021
Other Corp — Full-time Software Engineer | Jan 2021 - Dec 2022
PROJECTS
Personal Project — Developer | Jan 2020 - Dec 2022""", "R")
    assert parsed["combined_months_by_category"]["full_time_employment"] == 36
    assert parsed["combined_months_by_category"]["academic_or_personal_project"] == 36
    assert parsed["total_professional_months"] == 36


def test_skill_extraction_splits_labeled_durations():
    assert extract_skills("Python- Exprience - 24 months SQL- Experience - 12 months") == ["Python", "SQL"]


def test_skill_cleanup_preserves_dotnet_multword_and_drops_artifacts():
    raw = ".NET- Exprience - 24 months EXTRACT, TRANSFORM, AND LOAD- Exprience - 12 months months Azure- Exprience - 6 months Engineer- Exprience - 1 year"
    skills = extract_skills(raw)
    assert ".NET" in skills
    assert "EXTRACT, TRANSFORM, AND LOAD" in skills
    assert "months Azure" not in skills
    assert "Engineer" not in skills


def test_skill_cleanup_removes_section_headings_and_trailing_employer_text():
    raw = "Web Development: HTML5, CSS3, Bootstrap Company - Employer description - SQL"
    assert extract_skills(raw) == ["HTML5", "CSS3", "Bootstrap"]
    assert extract_skills("Web Development: HTML5, JavaScript. Database: MySQL. Development Tools: Notepad++") == ["HTML5", "JavaScript", "MySQL", "Notepad++"]


def test_project_extraction_requires_project_label():
    assert extract_projects("Project: Fraud Detection Tool Role: Developer Tools and Technologies: Python") == ["Fraud Detection Tool"]
    assert extract_projects("Built a tool at work") == []


def test_project_title_extraction_stops_before_description():
    text = "Diploma Project Name: VANET-virtual Ad Hoc Network Technology Used: Java. About Project: video streaming. Project Title: Fraud Detection Role: Analyst."
    assert extract_projects(text) == ["VANET-virtual Ad Hoc Network", "Fraud Detection"]


def test_project_extraction_ignores_client_project_and_status_headings():
    assert extract_projects("Client/Project: Barclays London Environment: Informatica") == []
    assert extract_projects("Project Status: 1) LIBS Brewery Completed 2) Citrus Processing Completed") == []


def test_project_extraction_handles_numbered_and_major_project_headings():
    text = "Major project: * Project based on AUTOMATIC WALL PLASTERING MACHINE. Project-I: Pragat Bharat Technologies Used: PHP."
    assert extract_projects(text) == ["AUTOMATIC WALL PLASTERING MACHINE", "Pragat Bharat"]


def test_unavailable_age_and_experience_are_na():
    record = structure_resume({"resume_id": "R1", "resume_text": "Graduated in 2020", "education_raw": "B.Tech CSE"})
    assert record["age"] == NA
    assert record["total_experience_years"] == NA


def test_counterfactual_changes_only_declared_attribute():
    original = {"resume_id": "R1", "college": "IIT Bombay", "skills": "Python", "projects": "P1"}
    variant, record = transform_counterfactual(original, "R1_CF_COLLEGE", "college", "BITS Pilani")
    assert variant["college"] == "BITS Pilani"
    assert {k: v for k, v in variant.items() if k != "college"} == {k: v for k, v in original.items() if k != "college"}
    assert record == {"original_resume_id": "R1", "counterfactual_id": "R1_CF_COLLEGE", "changed_attribute": "college", "original_value": "IIT Bombay", "counterfactual_value": "BITS Pilani"}


def test_validation_flags_duplicate_resume_ids_and_impossible_experience():
    resumes = pd.DataFrame([
        {"resume_id": "R1", "name": "A", "college": "X", "degree": NA, "course": NA, "resume_text": "", "total_experience_years": "100"},
        {"resume_id": "R1", "name": "B", "college": "Y", "degree": NA, "course": NA, "resume_text": "", "total_experience_years": NA},
    ])
    checks = set(validate_data(resumes)["check"])
    assert "duplicate_resume_id" in checks
    assert "impossible_experience" in checks


def test_validation_flags_unintended_counterfactual_changes():
    resumes = pd.DataFrame([{
        "resume_id": "R1", "name": "A", "college": "X", "degree": NA,
        "course": NA, "resume_text": "", "total_experience_years": NA,
    }])
    counterfactuals = pd.DataFrame([{
        "counterfactual_id": "CF1", "original_resume_id": "R1",
        "changed_attribute": "college", "counterfactual_value": "Y",
        "name": "B", "college": "Y", "degree": NA, "course": NA,
        "resume_text": "", "total_experience_years": NA,
    }])
    checks = set(validate_data(resumes, counterfactuals=counterfactuals)["check"])
    assert "unintended_counterfactual_change" in checks


def test_existing_company_course_mapping_is_project_data():
    mappings = pd.read_csv("data/company_course_preference_mapping.csv", keep_default_na=False)
    assert mappings["job_id"].nunique() == 25
    assert (mappings["mapping_status"] == "project_defined").all()
    assert (mappings["unrelated_courses"] == NA).all()
