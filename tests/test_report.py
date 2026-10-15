from pathlib import Path

from audit.report import PROJECT_ROOT, generate_report


def test_report_uses_current_intersectionality_schema():
    destination = PROJECT_ROOT / "tests" / ".audit_report_test.txt"
    try:
        generate_report(destination)
        report = destination.read_text(encoding="utf-8")

        assert "Joint cases: 650" in report
        assert "Interaction effect is the deviation" in report
        assert "Cases flipping at one or more supported thresholds" in report
        assert "does not establish discrimination" in report
    finally:
        Path(destination).unlink(missing_ok=True)

