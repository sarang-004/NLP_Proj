
from pathlib import Path
import pandas as pd

path = Path("screening_output_final/screening_results.csv").resolve()
print("Reading:", path)

results = pd.read_csv(path, encoding="utf-8-sig")
results.columns = results.columns.str.strip()

print("\nColumns found:")
print(results.columns.tolist())

score_candidates = ["final_score", "final score", "score", "finalscore"]
score_col = next(
    (c for c in score_candidates if c in results.columns),
    None
)

if score_col is None:
    print("\nNo recognized final-score column.")
    print("Possible score columns:", [
        c for c in results.columns if "score" in c.lower()
    ])
else:
    print("\nUsing score column:", score_col)

    results[score_col] = pd.to_numeric(
        results[score_col], errors="coerce"
    )

    if "job_id" not in results.columns:
        raise ValueError("CSV has no job_id column.")

    pairs = []
    for job_id, group in results.groupby("job_id"):
        ranked = group.dropna(subset=[score_col]).sort_values(
            score_col, ascending=False
        )
        if len(ranked) >= 2:
            pairs.append((
                abs(ranked.iloc[0][score_col] - ranked.iloc[1][score_col]),
                job_id, ranked.iloc[0], ranked.iloc[1]
            ))

    if not pairs:
        print("No job has at least two candidates with valid scores.")
    else:
        gap, job_id, a, b = min(pairs, key=lambda x: x[0])
        print(f"\nJob: {job_id}")
        print(f"Score gap: {gap:.6f}")

        fields = [
            "resume_id", "name", "rank", score_col,
            "skill_score", "experience_score", "education_score",
            "course_score", "education_course_score", "project_score",
            "semantic_score", "matched_skills", "missing_skills",
            "candidate_experience_years", "required_experience_years",
            "experience_status", "degree", "degree_status",
            "course", "course_status", "project_evidence",
            "semantic_evidence", "decision_note"
        ]

        for label, row in [
            ("Candidate A (higher score)", a),
            ("Candidate B (second highest)", b)
        ]:
            print(f"\n--- {label} ---")
            for field in fields:
                if field in results.columns:
                    print(f"{field}: {row.get(field, 'N/A')}")

