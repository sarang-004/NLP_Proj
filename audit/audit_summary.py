import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = Path("outputs/audit_signals.csv")
OUTPUT_FILE = Path("outputs/audit_summary.csv")


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print(f"Input rows: {len(df)}")


# ============================================================
# Summary by proxy type
# ============================================================

rows = []

for proxy, group in df.groupby("proxy_type"):

    abs_delta = group["abs_delta"]

    unsupported = group[
        group["context_group"] == "Contextually Unsupported"
    ]

    high_unsupported = unsupported[
        unsupported["magnitude_category"].isin(
            ["High", "Very High"]
        )
    ]

    rows.append({
        "proxy_type": proxy,

        "total_cases": len(group),

        "non_zero_changes": int(
            (abs_delta > 1e-9).sum()
        ),

        "no_material_change": int(
            (abs_delta <= 1e-9).sum()
        ),

        "mean_abs_delta": abs_delta.mean(),

        "median_abs_delta": abs_delta.median(),

        "p75_abs_delta": abs_delta.quantile(0.75),

        "p90_abs_delta": abs_delta.quantile(0.90),

        "p95_abs_delta": abs_delta.quantile(0.95),

        "max_abs_delta": abs_delta.max(),

        "score_increases": int(
            (group["score_delta"] > 1e-9).sum()
        ),

        "score_decreases": int(
            (group["score_delta"] < -1e-9).sum()
        ),

        "contextually_unsupported": len(
            unsupported
        ),

        "unsupported_high_or_very_high": len(
            high_unsupported
        ),

        "contextually_supported": int(
            (
                group["context_group"]
                == "Contextually Supported"
            ).sum()
        ),

        "requirement_related": int(
            (
                group["context_group"]
                == "Requirement Related"
            ).sum()
        ),
    })


summary = pd.DataFrame(rows)


# ============================================================
# Sort
# ============================================================

summary = summary.sort_values(
    "proxy_type"
).reset_index(drop=True)


# ============================================================
# Save
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Display
# ============================================================

print()
print("Summary by proxy type:")
print(
    summary.round(6).to_string(index=False)
)


# ============================================================
# Direct proxy summary
# ============================================================

direct = summary[
    summary["proxy_type"].isin(
        ["name", "age", "city", "college"]
    )
]

print()
print("Direct proxy summary:")
print(
    direct.round(6).to_string(index=False)
)


# ============================================================
# Experience summary
# ============================================================

experience = summary[
    summary["proxy_type"] == "experience"
]

print()
print("Experience summary:")
print(
    experience.round(6).to_string(index=False)
)


# ============================================================
# Qualification summary
# ============================================================

qualification = summary[
    summary["proxy_type"].isin(
        ["degree", "course"]
    )
]

print()
print("Qualification summary:")
print(
    qualification.round(6).to_string(index=False)
)


# ============================================================
# Intersectional summary
# ============================================================

intersectional = summary[
    summary["proxy_type"].str.contains(
        "+",
        regex=False
    )
]

print()
print("Intersectional transformations:")
print(
    intersectional.round(6).to_string(index=False)
)


# ============================================================
# Overall direct-proxy statistics
# ============================================================

direct_rows = df[
    df["proxy_type"].isin(
        ["name", "age", "city", "college"]
    )
]

print()
print("Overall direct-proxy statistics:")

print(
    f"Cases: {len(direct_rows)}"
)

print(
    f"Non-zero changes: "
    f"{(direct_rows['abs_delta'] > 1e-9).sum()}"
)

print(
    f"Mean |delta|: "
    f"{direct_rows['abs_delta'].mean():.6f}"
)

print(
    f"Median |delta|: "
    f"{direct_rows['abs_delta'].median():.6f}"
)

print(
    f"P95 |delta|: "
    f"{direct_rows['abs_delta'].quantile(.95):.6f}"
)

print(
    f"Maximum |delta|: "
    f"{direct_rows['abs_delta'].max():.6f}"
)

print(
    f"Unsupported cases: "
    f"{(
        direct_rows['context_group']
        == 'Contextually Unsupported'
    ).sum()}"
)

print(
    f"Unsupported High/Very High: "
    f"{len(
        direct_rows[
            (direct_rows['context_group']
             == 'Contextually Unsupported')
            &
            (direct_rows['magnitude_category']
             .isin(['High', 'Very High']))
        ]
    )}"
)


print()
print(f"Saved: {OUTPUT_FILE}")

print("=" * 70)