# NLP_Proj implementation status

Updated 2026-10-15. Source changes are implemented in the current workspace. Existing validated screening and audit datasets were not regenerated. The pre-existing `outputs/audit_report.txt` modification remains untouched during this implementation pass; the report test writes to and removes a temporary test file.

## Scoring methodology assessment (before this scoring revision)

- `run_screening.score_experience()` compared total reported career years to a JD minimum and returned `1.0` when no minimum was configured. It had no role- or technology-specific duration evidence.
- `run_screening.score_skills()` divided exact/alias keyword matches by every pipe-delimited skill; it did not retain source excerpts or separate a skills-list mention from work evidence. Uploaded JD terms were detected wherever they appeared, including responsibilities.
- `run_screening.score_education()` averaged degree and course results even when the JD supplied no criterion; a missing candidate field yielded zero. Missing/unconfigured factors other than projects still affected the denominator.
- `screening.decision_layer.evaluate()` exposed contributions but independently reconstructed weights/evidence, while `explainability/explainability.py` duplicated legacy component rules.
- New scores had no methodology version. Existing detailed counterfactual data has no trustworthy methodology-version field; it must remain legacy/unversioned and cannot be mixed with new scores without an explicit controlled rescore.

## Implemented

- New canonical scoring uses `evidence-v2` in `run_screening.score()`, shared by upload and batch decision scoring. The original component weights are unchanged. Requirement extraction/classification is implemented in `screening/requirements.py`; it preserves JD clauses, flexibility, responsibilities, duration criteria, and eligible skill/education/course context. JD skill terms use the project vocabulary and responsibility mentions are not automatically scored.
- Experience scoring distinguishes a total-years requirement from technology/activity-specific duration. Specific durations require explicit resume evidence; a project duration connected to a technology list and relevant activity is shown as ambiguous review evidence, not converted to tech-specific years. Employment-date extraction now accepts `MM.YYYY` and named-month periods, merges overlaps, and records the evaluation as-of date; a resume's profile-stated total is retained separately. Skill evidence distinguishes professional action evidence, a list mention, related aliases, and not demonstrated. No skill mention is interpreted as proof of absence.
- Absent JD factors and criterion information unavailable in the resume are excluded from the normalized denominator; explicit mismatches remain zero. Degree and course remain one configured component, with only specified/evidenced subcriteria included. This denominator behavior and the unchanged weights are documented in `SCORING_METHOD.md`.
- Upload results expose score methodology version, component scores, effective weights, weighted contributions, exact formula, rank, classified JD source statements, resume excerpts, experience context, and degree/course status. `/api/evaluate` accepts up to ten resumes with pasted JD or text-based PDF; uploads remain in memory. The supplied JD and resume PDFs are exercised through actual PDF extraction, the upload adapter, and the Flask endpoint in the optional source-document regression test.
- `resume_screening.py` public scoring functions now delegate to the shared upload/canonical scorer instead of producing a separate 70/30 score.
- `explainability/explainability.py` uses canonical decision output for `evidence-v2`; legacy output rows without a version remain on the legacy explanation route. Counterfactual, delta, case reconciliation, intersectionality, and unified-audit joins reject known version mismatches; existing datasets remain untouched and unversioned.
- Dashboard same-job comparison remains separate from scoring, with score/rank, component and weighted-contribution differences, and supporting evidence.
- `audit/case_validation.py` reconciles on `(resume_id, job_id, variant_id)`, checks proxy/value/score/delta agreement, and retains unmatched cases. Dashboard flip calculations use authoritative detailed score rows and configured thresholds rather than imported flip booleans.
- Current reconciliation yields 1,424 authoritative unique cases, 200 unmatched supplemental age rows, and 579 supplemental rows with score conflicts; only 21 of the 600 joinable supplemental rows agree on recorded scores. The 579 remain separately counted as supplemental conflicts and are not counted as additional cases. Threshold crossing counts are recalculated from authoritative records.
- `audit/context_analysis.py` returns evidence and requirement-aware classifications for the eight requested attributes and joint cases. Score change alone is not treated as proof of bias. Missing context and unknown mappings can require review.
- `audit/intersectionality.py` checks that individual and joint changes use the same candidate, job, and original/transformed values. It recalculates additive and interaction deltas only for compatible cases. Current detailed data has 650 joint rows; current transformations are compatible.
- `audit/fairness.py` calculates descriptive metrics at 0.55, 0.60, and 0.65. Dashboard calculations use structured resume data in memory rather than trusting stale fairness output CSVs or writing files. The module's explicit CLI still writes outputs when separately invoked.
- `templates/dashboard.html` integrates upload, scoring, comparisons, audit status/search, contextual evidence, intersectional case and summary views, fairness summaries, and threshold sensitivity.
- `audit/report.py` uses the current intersectionality schema and dashboard summaries, and qualifies the interpretation of score changes and descriptive fairness metrics.

## Verification

- `PYTHONPATH=. pytest -q`: **68 passed in 18.07s** after the final changes.
- `PYTHONPATH=. pytest -q tests/test_report.py`: 1 passed.
- `git diff --check`: passed; Git reported line-ending conversion notices only.
- Tests exercise requirement categories (including responsibilities and flexible qualifications), skill evidence levels, technology-specific duration, education/course missingness, denominator and contribution reproducibility, methodology versions, audit version compatibility, upload ranking, and dashboard/API integration.
- `tests/test_scoring_evidence_v2.py::test_actual_aws_engineer_and_resume_regression` passed against the two supplied PDF files. It verifies parser output, one-year Java QA automation context, no inferred Ruby/AWS development years, missing education/course exclusion, reproducible components, and the multipart Flask endpoint. It skips on another machine if those user-local source PDFs are absent.
- Production MiniLM inference/download was not run; scoring tests use a deterministic test embedding model. PDF extraction was exercised with the supplied PDFs. DOCX extraction remains covered only by the existing adapter path/tests if present; no actual DOCX source was supplied.
- `git diff --check`: passed. Git emitted only line-ending conversion notices for Windows text files.

## Remaining data and interpretation limits

- Current authoritative detailed scores do not contain age, experience, or employment-gap counterfactuals. The 200 supplemental age cases do not join to authoritative detailed scores and remain unmatched; no missing personal data or scores were fabricated.
- Resume/JD requirement extraction is deterministic and vocabulary/context based, not a full language-understanding model. Uncertain statements are retained for review; new or unusual skill names may not be recognized unless present in the project's structured JD vocabulary. Production embedding-model behavior was not executed in this audit.
- Technology-specific durations are credited only from explicit resume statements that associate duration, technology, and activity. A project duration and tool list may be surfaced for review but do not by themselves establish that the candidate used the technology throughout the project.
- Existing validated screening and audit output files were not regenerated. Their records do not carry a trusted version; dashboard audit analysis continues to show the existing legacy dataset, and new v2 rows must be generated through a controlled scoring run before they can replace it.
- Fairness results are descriptive score-threshold summaries, not observed hiring decisions or predictive validity. Available group attributes, missingness, and sample sizes constrain interpretation.
- No claim of discrimination or job suitability follows from score changes, threshold flips, group-rate differences, or interaction effects alone.

## Run

```powershell
$env:PYTHONPATH='.'
pytest -q
python app.py
```

For PDF/DOCX support, install dependencies with `pip install -r requirements.txt`.

