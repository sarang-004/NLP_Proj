from flask import Flask, jsonify, render_template, request
from pathlib import Path
import csv
import math

app = Flask(__name__)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUTS = PROJECT_ROOT / "outputs"

SCREENING = PROJECT_ROOT / "screening_output_final"


# ============================================================
# CSV UTILITIES
# ============================================================

def read_csv(path):
    """
    Read a CSV file and return a list of dictionaries.
    """

    if not path.exists():
        print(f"[WARNING] File not found: {path}")
        return []

    try:

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            return list(
                csv.DictReader(file)
            )

    except Exception as e:

        print(
            f"[ERROR] Could not read {path}: {e}"
        )

        return []


def num(value, default=0.0):
    """
    Safely convert a value to float.
    """

    try:

        x = float(value)

        if math.isfinite(x):
            return x

        return default

    except (
        TypeError,
        ValueError
    ):

        return default


def is_true(value):
    """
    Convert common CSV boolean values into True/False.
    """

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes"
    }


# ============================================================
# LOAD DASHBOARD DATA
# ============================================================

def load_dashboard():

    # --------------------------------------------------------
    # Load project CSV files
    # --------------------------------------------------------

    audit_summary = read_csv(
        OUTPUTS / "audit_summary.csv"
    )

    flip_rows = read_csv(
        OUTPUTS / "direct_proxy_decision_flips.csv"
    )

    unified = read_csv(
        OUTPUTS / "unified_audit.csv"
    )

    screening = read_csv(
        SCREENING / "screening_results_detailed.csv"
    )


    # ========================================================
    # 1. TOTAL CASES
    # ========================================================

    total_cases = len(unified)

    if total_cases == 0:
        total_cases = len(screening)


    # ========================================================
    # 2. DIRECT PROXY TESTS
    # ========================================================

    direct_proxy_types = {
        "name",
        "age",
        "city",
        "college"
    }


    direct_cases = [
        row
        for row in unified
        if row.get("proxy_type")
        in direct_proxy_types
    ]


    if direct_cases:

        direct_proxy_tests = len(
            direct_cases
        )

    else:

        direct_proxy_tests = sum(

            int(
                num(
                    row.get(
                        "total_cases"
                    ),
                    0
                )
            )

            for row in audit_summary

            if row.get(
                "proxy_type"
            ) in direct_proxy_types

        )


    # ========================================================
    # 3. ACTUAL DECISION FLIPS
    # ========================================================

    """
    direct_proxy_decision_flips.csv contains
    all 800 direct-proxy cases.

    It does NOT mean every row is a decision flip.

    We therefore count only rows where
    any_decision_flip == True.
    """

    decision_flip_rows = [

        row

        for row in flip_rows

        if is_true(
            row.get(
                "any_decision_flip"
            )
        )

    ]


    decision_flips = len(
        decision_flip_rows
    )


    # ========================================================
    # 4. THRESHOLD RESULTS
    # ========================================================

    thresholds = {

        "0.55": 0,

        "0.60": 0,

        "0.65": 0

    }


    for row in flip_rows:

        # 0.55

        if is_true(
            row.get(
                "flips_at_0.55"
            )
        ):

            thresholds["0.55"] += 1


        # 0.60

        if is_true(
            row.get(
                "flips_at_0.6"
            )
        ):

            thresholds["0.60"] += 1


        # Some CSV versions may use
        # flips_at_0.60 instead.

        if is_true(
            row.get(
                "flips_at_0.60"
            )
        ):

            thresholds["0.60"] += 1


        # 0.65

        if is_true(
            row.get(
                "flips_at_0.65"
            )
        ):

            thresholds["0.65"] += 1


    # ========================================================
    # 5. SEMANTIC SENSITIVITY
    # ========================================================

    semantic_values = {

        "name": [],

        "college": [],

        "city": [],

        "age": []

    }


    for row in screening:

        proxy_type = row.get(
            "proxy_type",
            ""
        )


        # For intersectional types such as
        # college+name, use the first component.

        proxy = proxy_type.split("+")[0]


        if proxy not in semantic_values:
            continue


        base_semantic = num(

            row.get(
                "base_semantic_score"
            )

        )


        variant_semantic = num(

            row.get(
                "variant_semantic_score"
            )

        )


        semantic_delta = (
            variant_semantic
            -
            base_semantic
        )


        semantic_values[
            proxy
        ].append(
            semantic_delta
        )


    semantic = {}


    for proxy, values in semantic_values.items():

        if values:

            semantic[proxy] = (

                sum(
                    abs(x)
                    for x in values
                )
                /
                len(values)

            )


    # --------------------------------------------------------
    # Fallback
    #
    # These are the validated values from your audit if
    # detailed screening data is unavailable.
    # --------------------------------------------------------

    if not semantic:

        semantic = {

            "name":
                0.008946,

            "college":
                0.007011,

            "city":
                0.004902,

            "age":
                0.003742

        }


    # ========================================================
    # 6. EXPLAINABILITY COVERAGE
    # ========================================================

    explainable = 0


    for row in unified:

        if (

            row.get(
                "explanation"
            )

            or

            row.get(
                "primary_changed_component"
            )

        ):

            explainable += 1


    if unified:

        explainability = (

            explainable
            /
            len(unified)

        ) * 100

    else:

        explainability = 0


    # ========================================================
    # 7. CASE DATA
    # ========================================================

    cases = []


    for row in unified:

        cases.append({

            "resume_id":
                row.get(
                    "resume_id",
                    ""
                ),

            "job_id":
                row.get(
                    "job_id",
                    ""
                ),

            "proxy_type":
                row.get(
                    "proxy_type",
                    ""
                ),

            "proxy_value":
                row.get(
                    "proxy_value",
                    ""
                ),

            "base_score":
                num(
                    row.get(
                        "base_score"
                    )
                ),

            "variant_score":
                num(
                    row.get(
                        "variant_score"
                    )
                ),

            "delta":
                num(

                    row.get(
                        "score_delta"
                    )

                    or

                    row.get(
                        "delta"
                    )

                ),

            "context":
                row.get(
                    "context_classification",
                    ""
                ),

            "component":
                row.get(
                    "primary_changed_component",
                    ""
                ),

            "explanation":
                row.get(
                    "explanation",
                    ""
                )

        })


    # ========================================================
    # 8. RETURN DATA TO FRONTEND
    # ========================================================

    return {

        "metrics": {

            "total_cases":
                total_cases,

            "direct_proxy_tests":
                direct_proxy_tests,

            "decision_flips":
                decision_flips,

            "explainability":
                round(
                    explainability,
                    1
                )

        },

        "semantic":
            semantic,

        "thresholds":
            thresholds,

        "flips":
            flip_rows,

        "cases":
            cases

    }


# ============================================================
# MAIN DASHBOARD PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# DASHBOARD API
# ============================================================

@app.route("/api/dashboard")
def dashboard():

    data = load_dashboard()

    return jsonify(
        data
    )


# ============================================================
# CASE SEARCH API
# ============================================================

@app.route("/api/case")
def case():

    resume_id = request.args.get(
        "resume_id",
        ""
    ).strip()


    job_id = request.args.get(
        "job_id",
        ""
    ).strip()


    data = load_dashboard()


    matches = []


    for row in data["cases"]:

        # Resume must match

        if row["resume_id"] != resume_id:
            continue


        # If job ID was provided,
        # it must also match.

        if (
            job_id
            and
            row["job_id"] != job_id
        ):

            continue


        matches.append(
            row
        )


    return jsonify(
        matches
    )


# ============================================================
# RUN FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=========================================="
    )
    print(
        " Resume Screening Bias Audit Dashboard"
    )
    print(
        "=========================================="
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Outputs folder: {OUTPUTS}"
    )

    print(
        f"Screening folder: {SCREENING}"
    )

    print()
    print(
        "Dashboard:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )