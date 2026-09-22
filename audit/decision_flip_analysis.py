import pandas as pd
from pathlib import Path


INPUT_FILE = Path("outputs/audit_signals.csv")
OUTPUT_FILE = Path("outputs/direct_proxy_decision_flips.csv")
SUMMARY_FILE = Path("outputs/direct_proxy_flip_summary.csv")


# Direct proxy attributes only
DIRECT_PROXIES = {
    "name",
    "age",
    "city",
    "college",
}


# Screening thresholds already used in the fairness analysis
THRESHOLDS = [0.55, 0.60, 0.65]


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_FILE)

df = df[
    df["proxy_type"].isin(DIRECT_PROXIES)
].copy()

print("=" * 70)
print("DIRECT PROXY DECISION-FLIP ANALYSIS")
print("=" * 70)

print(f"Direct-proxy cases: {len(df)}")


# ============================================================
# Decision flip calculation
# ============================================================

for threshold in THRESHOLDS:

    base_selected = df["base_score"] >= threshold
    variant_selected = df["variant_score"] >= threshold

    df[f"base_selected_{threshold}"] = base_selected
    df[f"variant_selected_{threshold}"] = variant_selected

    df[f"decision_flip_{threshold}"] = (
        base_selected != variant_selected
    )


# ============================================================
# Any threshold flip
# ============================================================

flip_columns = [
    f"decision_flip_{t}"
    for t in THRESHOLDS
]

df["any_decision_flip"] = df[
    flip_columns
].any(axis=1)


# ============================================================
# Flip direction
# ============================================================

def flip_direction(row):

    for threshold in THRESHOLDS:

        if not row[f"decision_flip_{threshold}"]:
            continue

        base = row[f"base_selected_{threshold}"]
        variant = row[f"variant_selected_{threshold}"]

        if not base and variant:
            return f"Positive flip @ {threshold}"

        if base and not variant:
            return f"Negative flip @ {threshold}"

    return "No Decision Flip"


df["flip_direction"] = df.apply(
    flip_direction,
    axis=1
)


# ============================================================
# Distance from threshold
# ============================================================

for threshold in THRESHOLDS:

    df[f"base_distance_{threshold}"] = (
        df["base_score"] - threshold
    ).abs()

    df[f"variant_distance_{threshold}"] = (
        df["variant_score"] - threshold
    ).abs()

    df[f"min_distance_{threshold}"] = df[
        [
            f"base_distance_{threshold}",
            f"variant_distance_{threshold}",
        ]
    ].min(axis=1)


# ============================================================
# High magnitude flag
# ============================================================

df["high_magnitude_direct_proxy"] = (
    df["magnitude_category"].isin(
        ["High", "Very High"]
    )
)

# ============================================================
# Absolute score delta
# ============================================================

df["abs_delta"] = df["score_delta"].abs()

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
    "magnitude_category",
    "audit_signal",
    "primary_changed_component",
    "high_magnitude_direct_proxy",
    "any_decision_flip",
    "flip_direction",
]

for threshold in THRESHOLDS:

    output_columns.extend([
        f"base_selected_{threshold}",
        f"variant_selected_{threshold}",
        f"decision_flip_{threshold}",
    ])


output = df[output_columns].copy()

output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Summary by proxy
# ============================================================

summary_rows = []

for proxy, group in df.groupby("proxy_type"):

    row = {
        "proxy_type": proxy,
        "total_cases": len(group),
        "high_magnitude_cases": int(
            group["high_magnitude_direct_proxy"].sum()
        ),
        "any_decision_flip": int(
            group["any_decision_flip"].sum()
        ),
    }

    for threshold in THRESHOLDS:

        row[f"flips_at_{threshold}"] = int(
            group[
                f"decision_flip_{threshold}"
            ].sum()
        )

        row[f"positive_flips_at_{threshold}"] = int(
            (
                group[f"decision_flip_{threshold}"]
                &
                ~group[f"base_selected_{threshold}"]
                &
                group[f"variant_selected_{threshold}"]
            ).sum()
        )

        row[f"negative_flips_at_{threshold}"] = int(
            (
                group[f"decision_flip_{threshold}"]
                &
                group[f"base_selected_{threshold}"]
                &
                ~group[f"variant_selected_{threshold}"]
            ).sum()
        )

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)


# Overall row
overall = {
    "proxy_type": "ALL_DIRECT_PROXIES",
    "total_cases": len(df),
    "high_magnitude_cases": int(
        df["high_magnitude_direct_proxy"].sum()
    ),
    "any_decision_flip": int(
        df["any_decision_flip"].sum()
    ),
}

for threshold in THRESHOLDS:

    overall[f"flips_at_{threshold}"] = int(
        df[f"decision_flip_{threshold}"].sum()
    )

    overall[f"positive_flips_at_{threshold}"] = int(
        (
            df[f"decision_flip_{threshold}"]
            &
            ~df[f"base_selected_{threshold}"]
            &
            df[f"variant_selected_{threshold}"]
        ).sum()
    )

    overall[f"negative_flips_at_{threshold}"] = int(
        (
            df[f"decision_flip_{threshold}"]
            &
            df[f"base_selected_{threshold}"]
            &
            ~df[f"variant_selected_{threshold}"]
        ).sum()
    )


summary = pd.concat(
    [
        summary,
        pd.DataFrame([overall])
    ],
    ignore_index=True
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# Print results
# ============================================================

print()
print("Decision flips by proxy:")
print(
    summary.to_string(index=False)
)


# ============================================================
# Print actual flipped cases
# ============================================================

flipped = output[
    output["any_decision_flip"]
].copy()

print()
print(
    f"Total cases with at least one decision flip: "
    f"{len(flipped)}"
)

if len(flipped) > 0:

    print()
    print("Decision-flip cases:")
    print(
    flipped[
        [
            "resume_id",
            "job_id",
            "proxy_type",
            "proxy_value",
            "base_score",
            "variant_score",
            "score_delta",
            "abs_delta",
            "context_classification",
            "magnitude_category",
            "flip_direction",
        ]
    ]
    .sort_values("abs_delta", ascending=False)
    .to_string(index=False)
)


print()
print(f"Saved: {OUTPUT_FILE}")
print(f"Saved: {SUMMARY_FILE}")

print("=" * 70)