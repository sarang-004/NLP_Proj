import pandas as pd

from audit.case_validation import reconcile_case_rows
from audit.context_analysis import classify_contextual_case
from audit.fairness import calculate_group_metrics, THRESHOLDS
from audit.intersectionality import validate_intersectionality


def test_supplemental_cases_are_verified_or_retained_as_unmatched():
    authoritative = [{"resume_id": "R1", "job_id": "J1", "variant_id": "V1", "proxy_type": "name", "base_score": 0.7, "variant_score": 0.6}]
    supplements = [
        {"resume_id": "R1", "job_id": "J1", "variant_id": "V1", "proxy_type": "name", "base_score": 0.7, "variant_score": 0.6},
        {"resume_id": "R1", "job_id": "J1", "variant_id": "V2", "proxy_type": "age", "base_score": 0.7, "variant_score": 0.65},
    ]
    result = reconcile_case_rows(authoritative, supplements)
    assert len(result["cases"]) == 2
    assert result["counts"]["verified_cases"] == 1
    assert result["counts"]["unmatched_cases"] == 1
    assert result["supplemental_validation"][1]["case_validation_status"] == "unmatched"


def test_name_score_change_is_flagged_for_review_as_potentially_biased():
    result = classify_contextual_case("name", "Asha Rao", "Lina Shah", 0.7, 0.68, {}, {"semantic": -0.02})
    assert result["classification"] == "Potentially biased"


def test_case_reconciliation_rejects_mixed_scoring_versions():
    from audit.case_validation import reconcile_case_rows

    source = [{"resume_id": "R", "job_id": "J", "variant_id": "V", "proxy_type": "age",
               "proxy_value": "31", "original_value": "30", "base_score": 0.5, "variant_score": 0.6,
               "base_score_methodology_version": "evidence-v2", "variant_score_methodology_version": "evidence-v2"}]
    supplement = [{**source[0], "base_score_methodology_version": "legacy-v1", "variant_score_methodology_version": "legacy-v1"}]
    result = reconcile_case_rows(source, supplement)
    assert result["supplemental_validation"][0]["case_validation_status"] == "invalid"
    assert "score_methodology_version" in result["supplemental_validation"][0]["case_validation_reason"]


def test_intersectionality_rejects_cross_version_component_cases():
    import pandas as pd
    from audit.intersectionality import validate_intersectionality

    frame = pd.DataFrame([
        {"resume_id": "R", "job_id": "J", "variant_id": "C", "proxy_type": "college", "original_value": "A", "counterfactual_value": "B", "base_score": .5, "variant_score": .6, "base_score_methodology_version": "legacy-v1"},
        {"resume_id": "R", "job_id": "J", "variant_id": "T", "proxy_type": "city", "original_value": "X", "counterfactual_value": "Y", "base_score": .5, "variant_score": .4, "base_score_methodology_version": "legacy-v1"},
        {"resume_id": "R", "job_id": "J", "variant_id": "CT", "proxy_type": "college+city", "original_value": "A|X", "counterfactual_value": "B|Y", "base_score": .5, "variant_score": .3, "base_score_methodology_version": "evidence-v2"},
    ])
    result = validate_intersectionality(frame)
    assert result.iloc[0]["match_status"] == "scoring_version_mismatch"


def test_unknown_course_mapping_and_joint_change_require_review():
    job = {"preferred_course": "Computer Science", "accepted_courses": "Computer Science|Information Technology"}
    course = classify_contextual_case("course", "Computer Science", "Unmapped Course", 0.8, 0.7, job, {"education_course": -0.1})
    joint = classify_contextual_case("name+city", "A|B", "C|D", 0.8, 0.7, job)
    assert course["classification"] == "Requires review"
    assert joint["classification"] == "Requires review"


def test_degree_and_experience_use_explicit_requirements():
    degree = classify_contextual_case("degree", "Diploma", "B.Tech", 0.5, 0.7, {"preferred_degree": "B.Tech", "accepted_degrees": "B.Tech|B.E."}, {"education_course": 0.2})
    experience = classify_contextual_case("experience", "12 months", "36 months", 0.5, 0.7, {"minimum_total_experience_years": 2}, {"experience": 0.2})
    missing = classify_contextual_case("experience", "NA", "36 months", 0.5, 0.7, {}, {})
    assert degree["classification"] == "Job-relevant"
    assert experience["classification"] == "Job-relevant"
    assert missing["classification"] == "Requires review"


def _intersection_frame(individual_city="Pune"):
    common = {"resume_id": "R1", "job_id": "J1", "base_score": 0.5}
    rows = [
        {**common, "variant_id": "city1", "proxy_type": "city", "original_value": "Mumbai", "counterfactual_value": individual_city, "variant_score": 0.48},
        {**common, "variant_id": "college1", "proxy_type": "college", "original_value": "College A", "counterfactual_value": "College B", "variant_score": 0.49},
        {**common, "variant_id": "joint1", "proxy_type": "college+city", "original_value": "College A|Mumbai", "counterfactual_value": "College B|Pune", "variant_score": 0.46},
    ]
    return pd.DataFrame(rows)


def test_intersectional_matching_checks_transformation_values():
    matched = validate_intersectionality(_intersection_frame())
    mismatch = validate_intersectionality(_intersection_frame(individual_city="Delhi"))
    assert matched.iloc[0].match_status == "matched"
    assert abs(matched.iloc[0].expected_additive_delta - (-0.03)) < 1e-9
    assert mismatch.iloc[0].match_status == "transformation_mismatch"
    assert pd.isna(mismatch.iloc[0].interaction_effect)


def test_fairness_uses_unique_candidate_denominator_and_supported_thresholds():
    base = pd.DataFrame({"resume_id": ["R1", "R2", "R3"], "score": [0.7, 0.55, 0.4], "group": ["A", "A", "B"], "selected": [1, 1, 0]})
    result = calculate_group_metrics(base, "test_group")
    assert result.total_candidates.sum() == 3
    assert result.selected_candidates.sum() == 2
    assert THRESHOLDS == [0.55, 0.60, 0.65]
