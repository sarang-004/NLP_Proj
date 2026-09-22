import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("outputs/unified_audit.csv")
OUTPUT_FILE = Path("outputs/audit_signals.csv")
THRESHOLD_FILE = Path("outputs/audit_thresholds.csv")


# Attributes whose changes are primarily being audited as
# potentially non-job-relevant proxies.
DIRECT_PROXY_FACTORS = {
    "name",
    "age",
    "city",
    "college",
}


# Experience is job-relevant, but can still be audited when
# its effect is not supported by the contextual analysis.
EXPERIENCE_FACTOR = "experience"


# Educational qualification attributes.
QUALIFICATION_FACTORS = {
    "degree",
    "course",
}


# ============================================================
# Load
# ============================================================

print("=" * 70)
print("FINAL AUDIT SIGNAL GENERATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Input rows: {len(df)}")


# ============================================================
# Validate
# ============================================================

required_columns = [
    "resume_id",
    "job_id",
    "variant_id",
    "proxy_type",
    "proxy_value",
    "base_score",
    "variant_score",
    "score_delta",
    "context_classification",
    "primary_changed_component",
    "explanation",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# Derived values
# ============================================================

df["abs_delta"] = df["score_delta"].abs()

df["direction"] = np.select(
    [
        df["score_delta"] > 1e-9,
        df["score_delta"] < -1e-9,
    ],
    [
        "Score Increased",
        "Score Decreased",
    ],
    default="No Change",
)


# ============================================================
# Context grouping
# ============================================================

def classify_context(context):

    context = str(context).strip()

    if context in {
        "Accepted",
        "Required",
        "Potentially Beneficial",
        "Strongly Beneficial",
        "Equivalent Acceptance",
    }:
        return "Contextually Supported"

    if context == "Requirement Not Met":
        return "Requirement Related"

    if context in {
        "Not Supported",
        "No Relevant Improvement",
    }:
        return "Contextually Unsupported"

    return "Other"


df["context_group"] = df["context_classification"].apply(
    classify_context
)


# ============================================================
# Factor-specific thresholds
# ============================================================
#
# Thresholds are calculated independently for each direct
# proxy. They describe unusually large changes within the
# current audit population.
#
# They are NOT universal fairness thresholds.
# ============================================================

thresholds = {}

for factor in DIRECT_PROXY_FACTORS:

    values = df.loc[
        df["proxy_type"] == factor,
        "abs_delta"
    ]

    if len(values) == 0:
        continue

    thresholds[factor] = {
        "p75": values.quantile(0.75),
        "p90": values.quantile(0.90),
        "p95": values.quantile(0.95),
        "p99": values.quantile(0.99),
    }


# ============================================================
# Magnitude classification
# ============================================================

def classify_magnitude(row):

    factor = row["proxy_type"]
    value = row["abs_delta"]

    # Intersectional transformations are analysed separately.
    if "+" in factor:
        return "Intersectional Change"

    # Qualification factors have their own interpretation.
    if factor in QUALIFICATION_FACTORS:
        return "Qualification Change"

    # Experience is handled as a job-relevant factor.
    if factor == EXPERIENCE_FACTOR:

        # Use the empirical distribution of experience effects.
        values = df.loc[
            df["proxy_type"] == EXPERIENCE_FACTOR,
            "abs_delta"
        ]

        p75 = values.quantile(0.75)
        p90 = values.quantile(0.90)
        p95 = values.quantile(0.95)

        if value <= p75:
            return "Low"

        elif value <= p90:
            return "Moderate"

        elif value <= p95:
            return "High"

        return "Very High"

    # Direct proxy factors.
    if factor not in thresholds:
        return "Unclassified"

    t = thresholds[factor]

    if value <= t["p75"]:
        return "Low"

    elif value <= t["p90"]:
        return "Moderate"

    elif value <= t["p95"]:
        return "High"

    return "Very High"


df["magnitude_category"] = df.apply(
    classify_magnitude,
    axis=1
)


# ============================================================
# Audit signal generation
# ============================================================

def generate_signal(row):

    factor = row["proxy_type"]
    context = row["context_group"]
    magnitude = row["magnitude_category"]
    abs_delta = row["abs_delta"]


    # --------------------------------------------------------
    # 1. No meaningful score change
    # --------------------------------------------------------

    if abs_delta <= 1e-9:
        return "No Material Change"


    # --------------------------------------------------------
    # 2. Intersectional transformations
    # --------------------------------------------------------
    #
    # Do NOT call these "unsupported".
    #
    # Their proper interpretation comes from the separate
    # intersectionality analysis and interaction_effect.
    # --------------------------------------------------------

    if "+" in factor:

        return "Intersectional Change — Separate Analysis"


    # --------------------------------------------------------
    # 3. Qualification transformations
    # --------------------------------------------------------

    if factor in QUALIFICATION_FACTORS:

        if context == "Contextually Supported":
            return "Qualification Change — Contextually Supported"

        if context == "Requirement Related":
            return "Qualification Change — Requirement Related"

        if context == "Contextually Unsupported":

            if magnitude in {"High", "Very High"}:
                return "Qualification Change — Unsupported High Magnitude"

            return "Qualification Change — Unsupported"

        return "Qualification Change — Review"


    # --------------------------------------------------------
    # 4. Experience
    # --------------------------------------------------------
    #
    # Experience is potentially job-relevant.
    # Therefore large changes are not automatically suspicious.
    # --------------------------------------------------------

    if factor == EXPERIENCE_FACTOR:

        if context == "Contextually Supported":
            return "Experience Change — Contextually Supported"

        if context == "Requirement Related":
            return "Experience Change — Requirement Related"

        if context == "Contextually Unsupported":

            if magnitude == "Very High":
                return "Potentially Unsupported Experience — Very High"

            if magnitude == "High":
                return "Potentially Unsupported Experience — High"

            if magnitude == "Moderate":
                return "Potentially Unsupported Experience — Moderate"

            return "Potentially Unsupported Experience — Low"

        return "Experience Change — Review"


    # --------------------------------------------------------
    # 5. Direct proxy attributes
    # --------------------------------------------------------
    #
    # Name, age, city and college are treated as direct
    # contextual proxies in this audit.
    # --------------------------------------------------------

    if factor in DIRECT_PROXY_FACTORS:

        if context == "Contextually Unsupported":

            if magnitude == "Very High":
                return "Potentially Unsupported Proxy — Very High"

            if magnitude == "High":
                return "Potentially Unsupported Proxy — High"

            if magnitude == "Moderate":
                return "Potentially Unsupported Proxy — Moderate"

            return "Potentially Unsupported Proxy — Low"


        if context == "Contextually Supported":
            return "Proxy Change — Contextually Supported"


        return "Proxy Change — Review"


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return "Review"


df["audit_signal"] = df.apply(
    generate_signal,
    axis=1
)


# ============================================================
# Threshold table
# ============================================================

threshold_rows = []

for factor, values in sorted(thresholds.items()):

    threshold_rows.append({
        "proxy_type": factor,
        "p75": values["p75"],
        "p90": values["p90"],
        "p95": values["p95"],
        "p99": values["p99"],
    })


# Add experience thresholds separately.
experience_values = df.loc[
    df["proxy_type"] == EXPERIENCE_FACTOR,
    "abs_delta"
]

if len(experience_values) > 0:

    threshold_rows.append({
        "proxy_type": EXPERIENCE_FACTOR,
        "p75": experience_values.quantile(0.75),
        "p90": experience_values.quantile(0.90),
        "p95": experience_values.quantile(0.95),
        "p99": experience_values.quantile(0.99),
    })


threshold_df = pd.DataFrame(threshold_rows)

threshold_df.to_csv(
    THRESHOLD_FILE,
    index=False
)


# ============================================================
# Final output
# ============================================================

output_columns = [
    "resume_id",
    "job_id",
    "variant_id",
    "proxy_type",
    "proxy_value",
    "base_score",
    "variant_score",
    "score_delta",
    "abs_delta",
    "direction",
    "context_classification",
    "context_group",
    "magnitude_category",
    "audit_signal",
    "primary_changed_component",
    "explanation",
]

output = df[output_columns].copy()

output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Reporting
# ============================================================

print()
print("Factor-specific thresholds:")
print(
    threshold_df
    .round(6)
    .to_string(index=False)
)

print()
print("Audit signal counts:")
print(
    output["audit_signal"]
    .value_counts()
    .to_string()
)

print()
print("Proxy-level signal counts:")
print(
    pd.crosstab(
        output["proxy_type"],
        output["audit_signal"]
    ).to_string()
)

print()
print("Context groups:")
print(
    output["context_group"]
    .value_counts()
    .to_string()
)

print()
print(f"Saved audit signals: {OUTPUT_FILE}")
print(f"Saved thresholds:    {THRESHOLD_FILE}")
print(f"Rows: {len(output)}")

print("=" * 70)