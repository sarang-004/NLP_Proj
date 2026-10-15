import pandas as pd

from app import app, load_dashboard


def test_dashboard_reconciles_case_counts_and_loads_analytics():
    data = load_dashboard()
    metrics = data["metrics"]
    assert metrics["total_cases"] == len(data["cases"])
    assert metrics["verified_cases"] == 1424
    assert metrics["unmatched_cases"] == 200
    assert metrics["invalid_cases"] == 0
    assert metrics["supplemental_conflicts"] == 579
    assert len(data["flips"]) == data["metrics"]["decision_flips"]
    assert len(data["intersectionality"]) == 650
    assert data["fairness"]
    assert data["fairness_sensitivity"]


def test_dashboard_endpoints_load_and_case_api_keeps_unmatched_rows_visible():
    client = app.test_client()
    assert client.get("/").status_code == 200
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["metrics"]["unmatched_cases"] == 200
    unmatched = next(row for row in payload["cases"] if row["case_validation_status"] == "unmatched")
    case_response = client.get("/api/case", query_string={"resume_id": unmatched["resume_id"], "job_id": unmatched["job_id"]})
    assert case_response.status_code == 200
    assert any(row["case_validation_status"] == "unmatched" for row in case_response.get_json())


def test_comparison_api_requires_same_job_and_two_candidates():
    client = app.test_client()
    base = pd.read_csv("screening_output_final/screening_decision_results.csv")
    job_id, pair = next((job, group) for job, group in base.groupby("job_id") if len(group) >= 2)
    response = client.get("/api/compare", query_string={"job_id": job_id, "resume_a": pair.iloc[0].resume_id, "resume_b": pair.iloc[1].resume_id})
    assert response.status_code == 200
    assert response.get_json()["candidate_a"]["job_id"] == response.get_json()["candidate_b"]["job_id"] == job_id
