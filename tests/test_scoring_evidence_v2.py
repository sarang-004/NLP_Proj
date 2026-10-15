from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from run_screening import WEIGHTS, score, score_education, score_experience, score_skills
from screening.requirements import METHOD_VERSION, UPLOAD_METHOD_VERSION, classify_jd_requirements, requirement_skill_names
from screening.upload_workflow import (evaluate_uploads, extract_uploaded_text, make_upload_job,
                                       structure_uploaded_resume, extract_preferred_qualification_evidence)


class FixedEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.tile(np.array([[1.0, 0.0]]), (len(texts), 1))


def test_requirement_classifier_keeps_context_duration_and_flexibility():
    parsed = classify_jd_requirements(
        "Our ideal qualifications may not all be met.\nQualifications\n"
        "4+ years developing in Ruby and Java\nExperience working with Git and Jenkins\n"
        "Responsibilities\nDevelop AWS services and monitor deployments"
    )
    assert parsed["jd_flexible_qualifications"] is True
    duration = [r for r in parsed["requirements"] if r["kind"] == "experience_duration"]
    assert {r["criterion"] for r in duration} == {"Ruby", "Java"}
    assert all(r["duration_years"] == 4 and r["category"] == "preferred" for r in duration)
    assert any(r["criterion"] == "Git" and r["kind"] == "skill" for r in parsed["requirements"])
    assert any(r["category"] == "responsibility" and "AWS" in r["source_text"] for r in parsed["requirements"])
    assert "AWS" not in requirement_skill_names({"requirements": parsed})


def test_explicit_and_uncertain_requirement_categories_and_responsibility_exclusion():
    explicit = classify_jd_requirements("Must have 5+ years using Python.")
    assert any(item["category"] == "explicit_minimum" for item in explicit["requirements"])
    uncertain = classify_jd_requirements("Experience working with Git and Jenkins.")
    assert all(item["category"] == "uncertain" for item in uncertain["requirements"] if item["kind"] == "skill")
    job = make_upload_job({"job_title": "Engineer", "job_description": "Responsibilities\n5+ years deploying AWS services"})
    assert job["minimum_total_experience_years"] == 0
    assert job["required_skills_list"] == []


def test_local_preference_does_not_downgrade_explicit_minimums():
    parsed = classify_jd_requirements(
        "Qualifications\nAt least 2 years of professional experience required.\n"
        "Python is required.\nA bachelor's degree is preferred."
    )
    experience = next(item for item in parsed["requirements"] if item["kind"] == "experience_duration")
    python = next(item for item in parsed["requirements"] if item["kind"] == "skill" and item["criterion"] == "Python")
    degree = next(item for item in parsed["requirements"] if item["kind"] == "education")
    assert experience["category"] == "explicit_minimum"
    assert python["category"] == "explicit_minimum"
    assert degree["category"] == "preferred"


def test_experience_requires_technology_and_activity_duration_evidence():
    job = pd.Series({
        "minimum_total_experience_years": 4,
        "requirements": classify_jd_requirements("Requires 4+ years developing in Java.")
    })
    qa_only = pd.Series({"total_experience_years_final": 7, "resume_text": "7+ years of QA experience. Java in skills."})
    assert score_experience(qa_only, job) is None
    supported = pd.Series({"total_experience_years_final": 7, "resume_text": "4 years developing Java applications."})
    assert score_experience(supported, job) == 1.0


def test_activity_scoring_uses_relevant_dated_roles_and_keeps_leadership_separate():
    jd = "At least 2 years of professional backend software development experience."
    job = make_upload_job({"job_title": "Backend Engineer", "job_description": jd})
    midhun_text = """EXPERIENCE
K7 Computing Pvt. Ltd., Chennai, India — Cybersecurity Intern | May 2025 – June 2025
Gained hands-on experience with cybersecurity tools.
Star Systems India Pvt. Ltd., India — AI & Machine Learning Intern | May 2026 – July 2026
Completed an internship focused on AI and Machine Learning."""
    gautham_text = """EXPERIENCE
Retailcloud — Software Engineering Intern
May 2026 – July 2026
Designed and built a backend application using Spring Boot.
CodeChef - VIT Student Chapter — Senior Core
March 2024 – February 2026
Developed frontend interfaces and co-organized a technical event."""
    midhun = structure_uploaded_resume(midhun_text, "Midhunlal P")
    gautham = structure_uploaded_resume(gautham_text, "Gautham Praveen")
    assert midhun["total_professional_experience_months"] == 5
    assert gautham["total_professional_experience_months"] == 3
    assert gautham["experience_months_by_category"]["student_leadership_or_extracurricular"] == 24
    assert score_experience(pd.Series(midhun), pd.Series(job)) is None
    assert score_experience(pd.Series(gautham), pd.Series(job)) == pytest.approx(3 / 24)

    candidates = evaluate_uploads([
        ("Midhunlal P", midhun_text),
        ("Gautham Praveen", gautham_text),
    ], job, FixedEmbeddingModel())
    by_name = {candidate["candidate_name"]: candidate for candidate in candidates}
    assert by_name["Gautham Praveen"]["experience"]["status"] == "Below requirement"
    assert by_name["Midhunlal P"]["experience"]["status"] == "Insufficient relevant role duration evidence"
    gautham_evidence = json.loads(by_name["Gautham Praveen"]["experience"]["evidence"])
    assert gautham_evidence["experience_requirement_type"] == "relevant_role"
    assert gautham_evidence["relevant_role_months"] == 3
    assert gautham_evidence["duration_ratio_numerator_months"] == 3
    assert gautham_evidence["duration_ratio_denominator_months"] == 24
    assert gautham_evidence["duration_ratio"] == pytest.approx(3 / 24)
    assert gautham_evidence["internship_months"] == 3
    assert any(item["experience_category"] == "student_leadership_or_extracurricular"
               for item in gautham_evidence["experience_records"])
    assert "3 months / 24 required months = 0.1250" in by_name["Gautham Praveen"]["explanation"]
    assert "student leadership, projects" in by_name["Gautham Praveen"]["explanation"]
    assert "no dated professional or internship role matched" in by_name["Midhunlal P"]["explanation"]
    midhun_evidence = json.loads(by_name["Midhunlal P"]["experience"]["evidence"])
    assert midhun_evidence["total_professional_months"] == 5
    assert midhun_evidence["internship_months"] == 5
    assert midhun_evidence["relevant_role_months"] is None


def test_software_development_duration_does_not_count_developed_understanding_as_software_work():
    job = make_upload_job({
        "job_title": "Junior Software Engineer",
        "job_description": "Minimum 1 year of professional software development experience.",
    })
    cybersecurity = structure_uploaded_resume("""EXPERIENCE
K7 Computing Pvt. Ltd. — Cybersecurity Intern | May 2025 - June 2025
Developed understanding of threat detection, malware analysis, and security protocols.
""", "Candidate")
    software_intern = structure_uploaded_resume("""EXPERIENCE
Retailcloud — Software Engineering Intern | May 2025 - July 2025
Designed and built backend services and Spring Boot microservices.
""", "Candidate")
    assert score_experience(pd.Series(cybersecurity), pd.Series(job)) is None
    assert score_experience(pd.Series(software_intern), pd.Series(job)) == pytest.approx(3 / 12)


def test_resume_projects_and_other_factors_remain_visible_without_jd_criteria():
    text = """EXPERIENCE
Acme — Full-time Software Engineer
Jan 2023 - Dec 2024
Built services and maintained production APIs.
EDUCATION
B.Tech in Computer Science
PROJECTS
Cloud File Synchronization Platform
Built a project to synchronize files between cloud storage providers.
"""
    job = make_upload_job({"job_title": "Software Engineer", "job_description": "Software engineer."})
    candidate = evaluate_uploads([("Candidate", text)], job, FixedEmbeddingModel())[0]
    assert candidate["component_scores"]["projects"] is not None
    assert "Cloud File Synchronization Platform" in candidate["project_evidence"]
    assert candidate["component_scores"]["experience"] is None
    assert candidate["experience"]["professional_months"] == 24
    assert candidate["experience"]["status"] == "Evidence only; JD has no duration requirement"
    assert candidate["degree_course_evidence"]["degree"] != "Not provided"
    assert candidate["component_scores"]["education_course"] is None
    assert candidate["degree_course_evidence"]["degree_status"] == "No JD requirement"
    assert "does not prove" in candidate["project_similarity_evidence"]
    assert "aggregate model signal" in candidate["semantic_evidence"]


def test_dashboard_uses_resume_evidence_language_instead_of_unavailable_for_empty_factors():
    template = Path("templates/dashboard.html").read_text(encoding="utf-8")
    assert "Resume experience evidence found" in template
    assert "No degree or course evidence found in the resume." in template
    assert "Project evidence found in the resume." in template
    assert "evidence remains visible even when the JD gives no basis" in template
    assert "Evidence completeness" in template
    assert "Skill-factor calculation" in template
    assert "Project score is cosine similarity" in template or "project_similarity_evidence" in template


def test_skill_evidence_distinguishes_listed_professional_and_absent():
    job = pd.Series({"required_skills": "Java|Ruby|AWS"})
    row = pd.Series({"skills_clean_new": "Java", "resume_text": "Skills: Java"})
    assert score_skills(row, job) == pytest.approx(1 / 6)
    evidence = {item["requirement"]: item for item in row["_skill_evidence"]}
    assert evidence["java"]["evidence_class"] == "listed_mention"
    assert evidence["ruby"]["evidence_class"] == "not_demonstrated"
    professional = pd.Series({"skills_clean_new": "Java", "resume_text": "I used Java to build an automation framework."})
    assert score_skills(professional, pd.Series({"required_skills": "Java"})) == 1.0
    assert professional["_skill_evidence"][0]["evidence_class"] == "professional_use_evidenced"
    related = pd.Series({"skills_clean_new": "MySQL", "resume_text": "Skills: MySQL"})
    assert score_skills(related, pd.Series({"required_skills": "SQL"})) == 0.5
    assert related["_skill_evidence"][0]["evidence_class"] == "related_transferable_skill"


def test_all_five_skill_criteria_expose_resume_section_and_evidence_type():
    requirements = "Python|SQL|REST API|Git|Automation"
    text = "SKILLS\nPython, SQL, REST API, Git, Automation"
    row = pd.Series({"skills_clean_new": requirements, "resume_text": text})
    assert score_skills(row, pd.Series({"required_skills": requirements})) == pytest.approx(0.5)
    assert len(row["_skill_evidence"]) == 5
    assert all(item["match_status"] == "matched" for item in row["_skill_evidence"])
    assert all(item["resume_section"] == "Skills" and item["evidence_type"] == "skills_list_mention"
               for item in row["_skill_evidence"])
    assert all(item["jd_duration_years"] is None and item["jd_proficiency_requirement"] is None
               for item in row["_skill_evidence"])


def test_skill_evidence_retains_jd_duration_and_proficiency_if_stated():
    jd = "Requires 4+ years developing in Python. Expert proficiency in SQL is required."
    job = make_upload_job({"job_title": "Engineer", "job_description": jd})
    row = pd.Series({"skills_clean_new": "Python|SQL", "resume_text": "Skills\nPython, SQL"})
    score_skills(row, pd.Series(job))
    evidence = {item["requirement"].casefold(): item for item in row["_skill_evidence"]}
    assert evidence["python"]["jd_duration_years"] == 4
    assert evidence["sql"]["jd_proficiency_requirement"] == "Expert proficiency in SQL is required."


def test_absent_education_course_criteria_and_missing_resume_information_are_unavailable():
    row = pd.Series({"primary_degree": "", "course_final": ""})
    no_criteria = pd.Series({"preferred_degree": "", "accepted_degrees": "", "preferred_course": "", "accepted_courses": ""})
    missing_resume = pd.Series({"preferred_degree": "B.Tech", "accepted_degrees": "B.Tech", "preferred_course": "", "accepted_courses": ""})
    explicit_mismatch = pd.Series({"preferred_degree": "B.Tech", "accepted_degrees": "B.Tech", "preferred_course": "", "accepted_courses": ""})
    assert score_education(row, no_criteria) is None
    assert score_education(row, missing_resume) is None
    assert score_education(pd.Series({"primary_degree": "Diploma", "course_final": ""}), explicit_mismatch) == 0.0


def test_jd_degree_and_course_are_extracted_only_when_stated():
    explicit = make_upload_job({"job_title": "Engineer", "job_description": "Qualifications: Bachelor's degree in Computer Science required."})
    assert explicit["accepted_degrees"]
    assert explicit["accepted_courses"] == ""
    absent = make_upload_job({"job_title": "Engineer", "job_description": "Qualifications: 3+ years of experience required."})
    assert absent["accepted_degrees"] == ""
    assert absent["accepted_courses"] == ""


def test_degree_subject_is_not_duplicated_as_course_and_coursework_is_not_skill():
    jd = (
        "Qualifications\nBachelor's degree in Computer Science is preferred.\n"
        "Coursework or certification in cloud computing is a plus."
    )
    parsed = classify_jd_requirements(jd)
    assert not any(item["kind"] == "course" and "degree in" in item["source_text"].lower()
                   for item in parsed["requirements"])
    assert "Cloud" not in requirement_skill_names({"requirements": parsed})
    assert "Cloud Computing" not in requirement_skill_names({"requirements": parsed})


def test_cloud_learning_plus_is_preferred_and_keeps_learning_types_distinct():
    jd = "Qualifications\nCoursework or certification in cloud computing is a plus."
    parsed = classify_jd_requirements(jd)
    preference = next(item for item in parsed["requirements"] if item["kind"] == "preferred_qualification")
    assert preference["criterion"].casefold() == "cloud computing"
    assert preference["category"] == preference["preference_status"] == "preferred"
    assert preference["evidence_types"] == ["coursework", "certification"]
    assert preference["source_text"] == "Coursework or certification in cloud computing is a plus."
    assert "Cloud" not in requirement_skill_names({"requirements": parsed})
    job = make_upload_job({"job_title": "Engineer", "job_description": jd})
    assert job["preferred_course"] == ""
    resume = ("CERTIFICATIONS\nAWS Certified Cloud Practitioner\nPROJECTS\nCloud synchronization application\n"
              "EXPERIENCE\nDeployed cloud services to production for Acme.")
    evidence = extract_preferred_qualification_evidence(resume, parsed["requirements"])
    assert [(item["evidence_type"], item["resume_section"]) for item in evidence] == [
        ("certification", "certifications"), ("project", "projects"),
        ("professional_experience", "professional experience")
    ]
    assert all(item["scored"] is False for item in evidence)
    assert not any(item["evidence_type"] == "coursework" for item in evidence)
    assert any(item["evidence_type"] == "professional_experience" for item in evidence)
    assert all(item["scored"] is False for item in evidence)


def test_preferred_language_variants_are_classified_without_making_mandatory():
    for wording in ("Cloud computing would be an advantage.", "Cloud exposure is desirable.",
                    "Familiarity with cloud platforms is preferred.", "Ideally, coursework in cloud computing."):
        parsed = classify_jd_requirements("Qualifications\n" + wording)
        assert all(item["category"] == "preferred" for item in parsed["requirements"])
        assert not any(item["category"] == "explicit_minimum" for item in parsed["requirements"])
    for wording in ("AWS certification is preferred.", "Cloud computing certification would be an advantage."):
        parsed = classify_jd_requirements("Qualifications\n" + wording)
        item = next(item for item in parsed["requirements"] if item["kind"] == "preferred_qualification")
        assert item["category"] == "preferred"
        assert "certification" in item["evidence_types"]


def test_common_plural_api_and_automated_testing_phrases_map_to_existing_skills():
    assert requirement_skill_names({"job_description": "Required skills: REST APIs and automated testing."}) == ["REST API", "Automation"]


def test_final_score_is_reproducible_and_versions_new_scores():
    job = pd.Series({
        "job_id": "J", "job_title": "QA automation", "required_skills": "Java|Ruby",
        "minimum_total_experience_years": 4,
        "requirements": classify_jd_requirements("Requires 4+ years developing in Ruby and Java."),
        "preferred_degree": "", "accepted_degrees": "", "preferred_course": "", "accepted_courses": "",
        "semantic_text_override": "Requires 4+ years developing in Ruby and Java.",
    })
    row = pd.Series({"resume_id": "R", "resume_text": "7 years QA experience. Skills: Java.",
                     "skills_clean_new": "Java", "total_experience_years_final": 7, "projects_extracted": ""})
    result = score(row, job, FixedEmbeddingModel())
    assert result["score_methodology_version"] == METHOD_VERSION
    assert result["experience_score"] is None
    assert result["education_course_score"] is None
    assert result["projects_score"] is None
    assert result["final_score"] == pytest.approx(sum(result["weighted_contributions"].values()))
    assert sum(result["effective_weights"].values()) == pytest.approx(1.0)
    available_components = {"skills": result["skills_score"], "experience": result["experience_score"],
                            "education_course": result["education_course_score"], "projects": result["projects_score"],
                            "semantic": result["semantic_score"]}
    assert result["score_denominator_weight"] == pytest.approx(sum(
        WEIGHTS[key] for key, value in available_components.items() if value is not None
    ))


def _education_job(text):
    return pd.Series(make_upload_job({"job_title": "Engineer", "job_description": text}))


@pytest.mark.parametrize("resume_text,expected_level,expected_subject,expected_specialization", [
    ("Education\nB.Tech in Computer Science with AI/ML specialization", "bachelor", "Computer Science", "AI/ML"),
    ("Education\nIntegrated M.Tech in Computer Science and Engineering with Data Science specialization", "integrated_master", "Computer Science and Engineering", "Data Science"),
])
def test_education_profile_separates_level_subject_and_specialization(resume_text, expected_level, expected_subject, expected_specialization):
    from data_layer import education_profile
    result = education_profile(resume_text)
    assert result["degree_level"] == expected_level
    assert result["degree_subject"] == expected_subject
    assert result["degree_specialization"] == expected_specialization


def test_integrated_mtech_matches_preferred_bachelors_subject_without_equating_levels():
    job = _education_job("A bachelor's degree in Computer Science or Software Engineering is preferred.")
    row = pd.Series(structure_uploaded_resume(
        "Education\nIntegrated M.Tech in Computer Science and Engineering with Data Science specialization", "R"))
    assert row["course_final"] == "Data Science"
    value = score_education(row, job)
    assert value == pytest.approx(0.8)
    assert row["_education_evidence"]["status"] == "preferred_subject_match_higher_level"
    assert row["_education_evidence"]["jd_preference"] == "preferred"
    assert row["_education_evidence"]["degree_level"] == "integrated_master"


def test_contextual_degree_match_is_preserved_when_course_override_is_supplied():
    job = pd.Series(make_upload_job({
        "job_title": "Engineer",
        "job_description": "A bachelor's degree in Computer Science is preferred.",
        "preferred_course": "Data Science",
    }))
    row = pd.Series(structure_uploaded_resume(
        "Education\nIntegrated M.Tech in Computer Science and Engineering with Data Science specialization", "R"))
    assert score_education(row, job) == pytest.approx(0.9)
    assert row["_education_evidence"]["score"] == pytest.approx(0.8)


def test_skill_jd_evidence_uses_best_scored_category_across_repeated_mentions():
    requirements = {"requirements": [
        {"kind": "skill", "criterion": "AWS", "category": "general_mention", "source_text": "AWS appears in company overview."},
        {"kind": "skill", "criterion": "AWS", "category": "preferred", "source_text": "AWS experience is preferred."},
        {"kind": "skill", "criterion": "AWS", "category": "responsibility", "source_text": "Build AWS services."},
    ]}
    job = pd.Series({"required_skills": "", "requirements": requirements})
    row = pd.Series({"skills_extracted": "AWS", "resume_text": "Skills\nAWS"})
    assert score_skills(row, job) == pytest.approx(0.5)
    aws = row["_skill_evidence"][0]
    assert aws["jd_requirement_category"] == "preferred"
    assert aws["jd_source_text"] == "AWS experience is preferred."


def test_upload_skill_policy_weights_required_above_preferred_but_audit_default_stays_v3():
    requirements = {"requirements": [
        {"kind": "skill", "criterion": "Python", "category": "explicit_minimum", "source_text": "Python required."},
        {"kind": "skill", "criterion": "Java", "category": "explicit_minimum", "source_text": "Java required."},
        {"kind": "skill", "criterion": "SQL", "category": "preferred", "source_text": "SQL preferred."},
    ]}
    job = pd.Series({"required_skills": "Python|Java|SQL", "requirements": requirements})
    row = pd.Series({"skills_extracted": "Python|Java|SQL",
                     "resume_text": "Skills\nPython, Java, SQL\nExperience\nI developed Python services."})
    legacy_score = score_skills(row, job)
    assert legacy_score == pytest.approx(2 / 3)
    upload_score = score_skills(row, job, category_weights={"explicit_minimum": 1.0, "preferred": 0.5})
    assert upload_score == pytest.approx(0.7)
    calculation = row["_skill_score_calculation"]
    assert calculation["weighted_evidence_total"] == pytest.approx(1.75)
    assert calculation["criterion_weight_total"] == pytest.approx(2.5)
    assert {item["requirement"]: item["criterion_weight"] for item in row["_skill_evidence"]} == {
        "java": 1.0, "python": 1.0, "sql": 0.5
    }


def test_upload_v4_does_not_change_default_audit_and_batch_score_path():
    from screening.decision_layer import UPLOAD_SKILL_CATEGORY_WEIGHTS

    job = pd.Series(make_upload_job({
        "job_title": "Engineer",
        "job_description": "Requirements\nPython is required.\nJava is preferred.",
    }))
    row = pd.Series(structure_uploaded_resume("Skills\nJava", "Candidate"))
    legacy = score(row.copy(), job.copy(), FixedEmbeddingModel())
    upload = score(row.copy(), job.copy(), FixedEmbeddingModel(),
                   skill_category_weights=UPLOAD_SKILL_CATEGORY_WEIGHTS)
    assert legacy["score_methodology_version"] == METHOD_VERSION == "evidence-v3"
    assert upload["score_methodology_version"] == UPLOAD_METHOD_VERSION == "evidence-v4"
    assert legacy["skills_score"] == pytest.approx(0.25)
    assert upload["skills_score"] == pytest.approx(1 / 6, abs=1e-6)


def test_general_and_responsibility_skill_mentions_are_not_scored_criteria():
    jd = classify_jd_requirements(
        "Qualifications\nPython is required.\nWe are interested in cloud operations.\n"
        "Responsibilities\nBuild and maintain AWS services."
    )
    assert requirement_skill_names({"requirements": jd}) == ["Python"]


@pytest.mark.parametrize(("jd", "resume", "expected"), [
    ("Requirements\nPython is required.\nJava is required.",
     "Experience\nUsed Python to build services. Used Java to implement APIs.", 1.0),
    ("Requirements\nPython is required.\nJava is required.",
     "Experience\nUsed Python to build services.", 0.5),
    ("Requirements\nPython is required.\nJava is preferred.",
     "Skills\nJava", 1 / 6),
    ("Qualifications\nPython is preferred.\nJava is preferred.",
     "Skills\nPython", 0.25),
])
def test_upload_skill_policy_required_preferred_coverage_cases(jd, resume, expected):
    from screening.decision_layer import UPLOAD_SKILL_CATEGORY_WEIGHTS

    job = make_upload_job({"job_title": "Engineer", "job_description": jd})
    row = pd.Series(structure_uploaded_resume(resume, "Candidate"))
    assert score_skills(row, pd.Series(job), category_weights=UPLOAD_SKILL_CATEGORY_WEIGHTS) == pytest.approx(expected)


def test_duplicate_skill_context_uses_explicit_requirement_and_does_not_double_count():
    requirements = {"requirements": [
        {"kind": "skill", "criterion": "Python", "category": "general_mention", "source_text": "Python overview."},
        {"kind": "skill", "criterion": "Python", "category": "preferred", "source_text": "Python preferred."},
        {"kind": "skill", "criterion": "Python", "category": "explicit_minimum", "source_text": "Python required."},
        {"kind": "skill", "criterion": "Python", "category": "responsibility", "source_text": "Maintain Python services."},
    ]}
    job = pd.Series({"required_skills": "", "requirements": requirements})
    row = pd.Series({"skills_extracted": "Python", "resume_text": "Skills: Python"})
    from screening.decision_layer import UPLOAD_SKILL_CATEGORY_WEIGHTS
    assert score_skills(row, job, category_weights=UPLOAD_SKILL_CATEGORY_WEIGHTS) == pytest.approx(0.5)
    assert len(row["_skill_evidence"]) == 1
    assert row["_skill_evidence"][0]["jd_requirement_category"] == "explicit_minimum"
    assert row["_skill_evidence"][0]["criterion_weight"] == pytest.approx(1.0)


def test_upload_evidence_completeness_keeps_absence_mismatch_and_unavailable_distinct():
    jd = ("Junior Engineer\nMinimum 1 year of professional backend software development experience.\n"
          "Required skills: Python.\nA bachelor's degree in Computer Science is preferred.\n"
          "Responsibilities\nDevelop backend software.")
    job = make_upload_job({"job_title": "Junior Engineer", "job_description": jd})
    resume = ("Education\nB.Tech in Mechanical Engineering\nExperience\n"
              "Cybersecurity Intern | May 2025 - June 2025\nAnalyzed security alerts.")
    candidate = evaluate_uploads([("Candidate", resume)], job, FixedEmbeddingModel())[0]
    completeness = candidate["evidence_completeness"]
    assert completeness["factor_status"]["skills"] == "not_demonstrated"
    assert completeness["factor_status"]["education_course"] == "explicitly_mismatched"
    assert completeness["factor_status"]["experience"] == "unavailable"
    assert completeness["factor_status"]["projects"] == "unavailable"
    assert completeness["factor_status"]["semantic"] == "aggregate_signal_available"
    assert completeness["criterion_evidence_coverage"] == pytest.approx(0.3333)
    assert "not proof the candidate lacks" in completeness["coverage_note"]
    assert candidate["score_methodology_version"] == UPLOAD_METHOD_VERSION


def test_matching_subject_with_different_bachelors_degree_title_scores_by_level_and_subject():
    job = _education_job("A bachelor's degree in Computer Science is preferred.")
    row = pd.Series(structure_uploaded_resume("Education\nB.Sc in Computer Science", "R"))
    assert score_education(row, job) == pytest.approx(1.0)
    assert row["_education_evidence"]["status"] == "required_level_and_subject_match"


def test_mandatory_bachelors_is_not_inferred_from_masters_and_missing_education_is_unavailable():
    job = _education_job("Bachelor's degree in Computer Science is required.")
    master = pd.Series(structure_uploaded_resume("Education\nM.Tech in Computer Science", "R"))
    assert score_education(master, job) is None
    assert master["_education_evidence"]["status"] == "mandatory_level_unverified"
    missing = pd.Series(structure_uploaded_resume("Experience\nEngineer", "R2"))
    assert score_education(missing, job) is None
    assert missing["_education_evidence"]["status"] == "missing_candidate_education"


def test_mandatory_bachelors_can_be_verified_when_separately_listed_with_matching_subject():
    job = _education_job("Bachelor's degree in Computer Science is required.")
    row = pd.Series(structure_uploaded_resume(
        "Education\nB.Sc in Computer Science\nIntegrated M.Tech in Computer Science and Engineering", "R"))
    assert score_education(row, job) == pytest.approx(1.0)
    assert row["_education_evidence"]["status"] == "mandatory_bachelors_explicitly_evidenced"
    assert row["_education_evidence"]["matched_credential"]["degree_level"] == "bachelor"


def test_explicit_education_subject_mismatch_scores_zero_and_preference_is_retained():
    job = _education_job("A bachelor's degree in Computer Science is preferred.")
    row = pd.Series(structure_uploaded_resume("Education\nB.Tech in Mechanical Engineering", "R"))
    assert score_education(row, job) == 0.0
    assert row["_education_evidence"]["status"] == "explicit_subject_mismatch"
    assert row["_education_evidence"]["jd_preference"] == "preferred"


def test_candidates_with_unavailable_experience_use_different_normalized_weights():
    job = pd.Series({"job_id": "J", "job_title": "Engineer", "required_skills": "Python",
                     "minimum_total_experience_years": 1, "requirements": classify_jd_requirements("Requires 1 year experience."),
                     "preferred_degree": "", "accepted_degrees": "", "preferred_course": "", "accepted_courses": "",
                     "semantic_text_override": "Engineer using Python", "job_description": "Requires 1 year experience."})
    with_experience = pd.Series({"resume_id": "A", "resume_text": "Python developer", "skills_clean_new": "Python",
                                 "total_experience_years_final": 2, "projects_extracted": "Python project"})
    without_experience = pd.Series({"resume_id": "B", "resume_text": "Python developer", "skills_clean_new": "Python",
                                    "projects_extracted": "Python project"})
    a = score(with_experience, job.copy(), FixedEmbeddingModel())
    b = score(without_experience, job.copy(), FixedEmbeddingModel())
    assert a["experience_score"] is not None
    assert b["experience_score"] is None
    assert a["score_denominator_weight"] == pytest.approx(0.85)
    assert b["score_denominator_weight"] == pytest.approx(0.65)
    assert a["effective_weights"]["skills"] == pytest.approx(WEIGHTS["skills"] / 0.85)
    assert b["effective_weights"]["skills"] == pytest.approx(WEIGHTS["skills"] / 0.65)
    for result in (a, b):
        assert result["final_score"] == pytest.approx(sum(result["weighted_contributions"].values()))


def test_supplied_cs_resumes_score_preferred_education_with_level_evidence(monkeypatch):
    gautham_path = Path(r"C:\Users\Sarang\Downloads\RESUME-GAUTHAM PRAVEEN.pdf")
    midhun_path = Path(r"C:\Users\Sarang\Downloads\MidhunlalResume.pdf")
    if not gautham_path.exists() or not midhun_path.exists():
        pytest.skip("Supplied candidate PDFs are not present; upload-source regression is optional outside the user's machine.")
    jd = ("Junior Software Engineer\nMinimum 1 year of professional software development experience.\n"
          "Required skills: Python, Java, SQL, Git, REST APIs.\n"
          "A bachelor's degree in Computer Science or Software Engineering is preferred.\n"
          "Responsibilities include developing and maintaining software applications, backend services, and REST APIs.")
    job = make_upload_job({"job_title": "Junior Software Engineer", "job_description": jd})
    uploads = []
    for path in (gautham_path, midhun_path):
        uploads.append((path.stem, extract_uploaded_text(path.name, path.open("rb")), path.name))
    candidates = evaluate_uploads(uploads, job, FixedEmbeddingModel())
    by_name = {item["candidate_name"].upper(): item for item in candidates}
    gautham = next(item for name, item in by_name.items() if "GAUTHAM" in name)
    midhun = next(item for name, item in by_name.items() if "MIDHUN" in name)
    assert gautham["degree_course_evidence"]["profile"]["degree_subject"] in {"Computer Science", "Computer Science and Engineering"}
    assert gautham["degree_course_evidence"]["profile"]["degree_specialization"].casefold() == "ai/ml"
    assert len(gautham["degree_course_evidence"]["profile"]["degree_evidence"]) <= 240
    assert "EXPERIENCE" not in gautham["degree_course_evidence"]["profile"]["degree_evidence"].upper()
    assert len(gautham["degree_course_evidence"]["profile"]["degree_credentials"]) == 1
    assert gautham["degree_course_evidence"]["degree_score"] > 0
    assert midhun["degree_course_evidence"]["profile"]["degree_level"] == "integrated_master"
    assert midhun["degree_course_evidence"]["profile"]["degree_subject"] == "Computer Science and Engineering"
    assert midhun["degree_course_evidence"]["profile"]["degree_specialization"].casefold() == "data science"
    assert len(midhun["degree_course_evidence"]["profile"]["degree_evidence"]) <= 240
    assert "SKILLS" not in midhun["degree_course_evidence"]["profile"]["degree_evidence"].upper()
    assert len(midhun["degree_course_evidence"]["profile"]["degree_credentials"]) == 1
    assert midhun["degree_course_evidence"]["profile"]["jd_preference"] == "preferred"
    assert midhun["degree_course_evidence"]["degree_score"] > 0
    assert gautham["effective_weights"] != midhun["effective_weights"]
    assert "experience" in midhun["excluded_factors"]

    import app as dashboard_app
    from io import BytesIO
    monkeypatch.setattr(dashboard_app, "_load_upload_model", lambda: FixedEmbeddingModel())
    response = dashboard_app.app.test_client().post("/api/evaluate", data={
        "jd_source": "text", "job_title": "Junior Software Engineer", "job_description": jd,
        "resumes": [(BytesIO(gautham_path.read_bytes()), gautham_path.name),
                    (BytesIO(midhun_path.read_bytes()), midhun_path.name)],
    }, content_type="multipart/form-data")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["score_comparability"]["effective_weights_differ"] is True
    assert "not directly comparable" in payload["score_comparability"]["warning"]
    assert payload["scoring_methodology_version"] == UPLOAD_METHOD_VERSION


def test_delta_pipeline_rejects_cross_version_scores(tmp_path):
    from audit.delta import calculate_deltas

    source = tmp_path / "scores.csv"
    target = tmp_path / "deltas.csv"
    pd.DataFrame([{
        "resume_id": "R", "job_id": "J", "variant_id": "V", "proxy_type": "age", "proxy_value": "30",
        "base_score": 0.5, "variant_score": 0.6,
        "base_score_methodology_version": "legacy-v1", "variant_score_methodology_version": METHOD_VERSION,
    }]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="across scoring methodology versions"):
        calculate_deltas(source, target)


def test_delta_pipeline_labels_existing_rows_legacy_and_accepts_same_version(tmp_path):
    from audit.delta import calculate_deltas

    source = tmp_path / "same-version.csv"
    target = tmp_path / "deltas.csv"
    base = {"resume_id": "R", "job_id": "J", "variant_id": "V", "proxy_type": "age", "proxy_value": "31", "base_score": 0.5, "variant_score": 0.6}
    pd.DataFrame([{**base, "base_score_methodology_version": METHOD_VERSION, "variant_score_methodology_version": METHOD_VERSION}]).to_csv(source, index=False)
    calculate_deltas(source, target)
    assert pd.read_csv(target).iloc[0]["score_methodology_version"] == METHOD_VERSION
    pd.DataFrame([base]).to_csv(source, index=False)
    calculate_deltas(source, target)
    assert pd.read_csv(target).iloc[0]["score_methodology_version"] == "legacy_unversioned"


def test_actual_aws_engineer_and_resume_regression(monkeypatch):
    jd_path = Path(r"C:\Users\Sarang\Downloads\aws-engineer-v1.pdf")
    resume_path = Path(r"C:\Users\Sarang\Downloads\Mobile-Application-Testing-Resume.pdf")
    if not jd_path.exists() or not resume_path.exists():
        pytest.skip("Supplied source PDFs are not present on this machine; source-document regression is optional outside the user's workspace.")
    jd = extract_uploaded_text(jd_path.name, jd_path.open("rb"))
    resume = extract_uploaded_text(resume_path.name, resume_path.open("rb"))
    job = make_upload_job({"job_title": "AWS Engineer", "job_description": jd})
    structured_resume = structure_uploaded_resume(resume, "Candidate 1")
    resume_skills = {item.casefold() for item in structured_resume["skills_extracted"].split("|")}
    assert "java" in resume_skills and any("automation" in item for item in resume_skills)
    candidates = evaluate_uploads([("Candidate 1", resume)], job, FixedEmbeddingModel())
    candidate = candidates[0]
    assert float(candidate["experience"]["years"]) >= 7.0
    assert candidate["experience"]["reported_years"] == 7.0
    assert candidate["experience"]["source"] == "explicit_profile_summary"
    assert "7+ years" in resume
    assert candidate["component_scores"]["experience"] is None
    assert candidate["experience"]["status"] == "Insufficient technology-specific duration evidence"
    tech_duration_evidence = json.loads(candidate["experience"]["evidence"])["technology_specific"]
    java_context = next(item for item in tech_duration_evidence if item["technology"].casefold() == "java")
    assert java_context["status"] == "contextual_project_duration_requires_review"
    assert java_context["approximate_project_duration_years"] == 1
    evidence = {item["requirement"].casefold(): item for item in candidate["skill_evidence"]}
    assert evidence["java"]["evidence_class"] in {"listed_mention", "professional_use_evidenced"}
    assert evidence["ruby"]["evidence_class"] == "not_demonstrated"
    assert evidence["aws"]["evidence_class"] == "not_demonstrated"
    assert evidence["git"]["evidence_class"] in {"listed_mention", "professional_use_evidenced"}
    assert evidence["jenkins"]["evidence_class"] in {"listed_mention", "professional_use_evidenced"}
    assert candidate["degree_course_evidence"]["degree_status"] == "No JD requirement"
    assert candidate["degree_course_evidence"]["course_status"] == "No JD requirement"
    assert "not demonstrated" in candidate["explanation"].casefold()
    assert candidate["final_score"] == pytest.approx(sum(candidate["weighted_contributions"].values()))
    assert "Ruby" in job["required_skills_list"] and "AWS" in job["required_skills_list"]

    import app as dashboard_app
    from io import BytesIO
    monkeypatch.setattr(dashboard_app, "_load_upload_model", lambda: FixedEmbeddingModel())
    response = dashboard_app.app.test_client().post("/api/evaluate", data={
        "jd_source": "pdf", "job_title": "AWS Engineer",
        "job_description_pdf": (BytesIO(jd_path.read_bytes()), jd_path.name),
        "resumes": (BytesIO(resume_path.read_bytes()), resume_path.name),
    }, content_type="multipart/form-data")
    assert response.status_code == 200
    api_result = response.get_json()
    assert api_result["scoring_methodology_version"] == UPLOAD_METHOD_VERSION
    assert api_result["candidates"][0]["component_scores"]["experience"] is None
