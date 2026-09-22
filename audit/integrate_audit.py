import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

SCREENING_FILE = (
    BASE_DIR
    / "screening_output_final"
    / "screening_results_detailed.csv"
)

CONTEXT_FILE = (
    BASE_DIR
    / "outputs"
    / "context_results.csv"
)

EXPLAINABILITY_FILE = (
    BASE_DIR
    / "outputs"
    / "explainability"
    / "counterfactual_explanations.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "unified_audit.csv"
)


# --------------------------------------------------
# Load
# --------------------------------------------------

print("Loading audit datasets...")

screening = pd.read_csv(SCREENING_FILE)
context = pd.read_csv(CONTEXT_FILE)
explainability = pd.read_csv(EXPLAINABILITY_FILE)

print(f"Screening rows:       {len(screening)}")
print(f"Context rows:         {len(context)}")
print(f"Explainability rows:  {len(explainability)}")


# --------------------------------------------------
# Join keys
# --------------------------------------------------

KEYS = [
    "resume_id",
    "job_id",
    "variant_id",
]


# --------------------------------------------------
# Validate uniqueness
# --------------------------------------------------

for name, df in [
    ("screening", screening),
    ("context", context),
    ("explainability", explainability),
]:
    duplicate_count = df.duplicated(KEYS).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{name} contains {duplicate_count} duplicate rows "
            f"for the keys {KEYS}"
        )


# --------------------------------------------------
# Select useful screening columns
# --------------------------------------------------

screening_columns = [
    "resume_id",
    "job_id",
    "variant_id",
    "proxy_type",
    "proxy_value",

    "base_skills_score",
    "base_experience_score",
    "base_education_course_score",
    "base_projects_score",
    "base_semantic_score",
    "base_score",

    "variant_skills_score",
    "variant_experience_score",
    "variant_education_course_score",
    "variant_projects_score",
    "variant_semantic_score",
    "variant_score",

    "score_delta",
]

screening = screening[screening_columns]


# --------------------------------------------------
# Select useful context columns
# --------------------------------------------------

context_columns = [
    "resume_id",
    "job_id",
    "variant_id",
    "company",
    "original_value",
    "variant_value",
    "original_course_status",
    "variant_course_status",
    "jd_requirement",
    "context_classification",
]

context = context[context_columns]


# --------------------------------------------------
# Select useful explainability columns
# --------------------------------------------------

explainability_columns = [
    "resume_id",
    "job_id",
    "variant_id",
    "primary_changed_component",
    "primary_component_change",
    "primary_weighted_change",
    "explanation",
]

explainability = explainability[explainability_columns]


# --------------------------------------------------
# Merge
# --------------------------------------------------

print("Merging screening + context...")

unified = screening.merge(
    context,
    on=KEYS,
    how="left",
    validate="one_to_one",
)

print("Merging explainability...")

unified = unified.merge(
    explainability,
    on=KEYS,
    how="left",
    validate="one_to_one",
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

print("\nValidating unified dataset...")

if len(unified) != len(screening):
    raise ValueError(
        "Unified row count does not match screening row count."
    )

missing_context = unified["context_classification"].isna().sum()
missing_explanation = unified["explanation"].isna().sum()

print(f"Unified rows: {len(unified)}")
print(f"Missing context rows: {missing_context}")
print(f"Missing explanation rows: {missing_explanation}")


# --------------------------------------------------
# Score consistency checks
# --------------------------------------------------

score_difference = (
    unified["variant_score"] - unified["base_score"]
)

delta_difference = (
    score_difference - unified["score_delta"]
).abs()

max_delta_difference = delta_difference.max()

print(
    f"Maximum score/delta inconsistency: "
    f"{max_delta_difference:.10f}"
)

if max_delta_difference > 1e-6:
    raise ValueError(
        "score_delta does not match variant_score - base_score."
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

unified.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved unified audit dataset to:")
print(OUTPUT_FILE)