"""Read-only reconciliation of supplemental audit rows to scored cases."""
from __future__ import annotations

KEYS = ("resume_id", "job_id", "variant_id")


def _key(row):
    return tuple(str(row.get(column, "")).strip() for column in KEYS)


def reconcile_case_rows(authoritative_rows, supplemental_rows, tolerance=1e-8):
    """Return authoritative cases plus unmatched/invalid supplemental rows.

    Supplemental scores are accepted only when IDs, proxy, and recorded scores
    agree with the authoritative detailed scoring output. No rows are dropped.
    """
    authoritative = {}
    for row in authoritative_rows:
        key = _key(row)
        authoritative.setdefault(key, []).append(row)
    cases = []
    case_by_key = {}
    for row in authoritative_rows:
        case = dict(row)
        key = _key(row)
        case["case_validation_status"] = "verified" if len(authoritative[key]) == 1 and all(key) else "invalid"
        if case["case_validation_status"] == "invalid":
            case["case_validation_reason"] = "Duplicate or incomplete authoritative case key."
        else:
            case["case_validation_reason"] = "Present in authoritative detailed screening output."
        cases.append(case)
        case_by_key.setdefault(key, []).append(case)

    supplemental_validation = []
    for supplement in supplemental_rows:
        key = _key(supplement)
        matches = authoritative.get(key, [])
        case = dict(supplement)
        if not all(key):
            status, reason = "invalid", "Supplemental row has an incomplete resume/job/variant key."
        elif not matches:
            status, reason = "unmatched", "No authoritative detailed screening row has this resume/job/variant key."
        elif len(matches) != 1:
            status, reason = "invalid", "Multiple authoritative rows match this supplemental case key."
        else:
            source = matches[0]
            mismatches = []
            version_fields = ("base_score_methodology_version", "variant_score_methodology_version")
            if any(field in supplement or field in source for field in version_fields):
                source_base = str(source.get(version_fields[0], "")).strip()
                source_variant = str(source.get(version_fields[1], "")).strip()
                if not source_base or not source_variant or source_base != source_variant:
                    mismatches.append("source_score_methodology_version")
                for field in version_fields:
                    external_version = str(supplement.get(field, "")).strip()
                    source_version = str(source.get(field, "")).strip()
                    if not external_version or not source_version or external_version != source_version:
                        mismatches.append("score_methodology_version")
                        break
            if str(supplement.get("proxy_type", "")).strip() != str(source.get("proxy_type", "")).strip():
                mismatches.append("proxy_type")
            for field, source_field in (("proxy_value", "proxy_value"), ("original_value", "original_value")):
                external_value = str(supplement.get(field, "")).strip()
                authoritative_value = str(source.get(source_field, "")).strip()
                if external_value and authoritative_value and external_value.casefold() != authoritative_value.casefold():
                    mismatches.append(field)
            for field in ("base_score", "variant_score"):
                try:
                    if abs(float(supplement[field]) - float(source[field])) > tolerance:
                        mismatches.append(field)
                except (KeyError, TypeError, ValueError):
                    mismatches.append(field)
            try:
                external_delta = float(supplement["variant_score"]) - float(supplement["base_score"])
                source_delta = float(source["variant_score"]) - float(source["base_score"])
                if abs(external_delta - source_delta) > tolerance:
                    mismatches.append("score_delta")
            except (KeyError, TypeError, ValueError):
                mismatches.append("score_delta")
            if mismatches:
                status, reason = "invalid", "Supplemental values disagree with authoritative fields: " + ", ".join(mismatches)
            else:
                status, reason = "verified", "Supplemental row agrees with authoritative IDs, proxy, and scores."
        case["case_validation_status"] = status
        case["case_validation_reason"] = reason
        supplemental_validation.append(case)
        if status == "verified":
            for authoritative_case in case_by_key.get(key, []):
                authoritative_case["supplemental_validation_status"] = "verified"
                authoritative_case["supplemental_validation_reason"] = reason
        elif status == "invalid" and len(matches) == 1:
            # Keep one canonical case, but expose that the supplemental
            # artifact conflicts with its authoritative score row.
            matches[0]["supplemental_validation_status"] = "invalid"
            matches[0]["supplemental_validation_reason"] = reason
        else:
            cases.append(case)
    counts = {
        "verified_cases": sum(row["case_validation_status"] == "verified" for row in cases),
        "unmatched_cases": sum(row["case_validation_status"] == "unmatched" for row in cases),
        "invalid_cases": sum(row["case_validation_status"] == "invalid" for row in cases),
        "supplemental_invalid_records": sum(row["case_validation_status"] == "invalid" for row in supplemental_validation),
        "unique_cases": len(cases),
    }
    return {"cases": cases, "supplemental_validation": supplemental_validation, "counts": counts}
