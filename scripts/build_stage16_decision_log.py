"""
scripts/build_stage16_decision_log.py
-------------------------------------
STAGE 16: Build Golden Evaluation Set Decision Log Artifacts

CLI script that builds, validates, and exports:
  - reports/stage16/golden_decision_log.json
  - reports/stage16/golden_decision_log.md
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.golden_decision_log import (
    build_golden_decision_log,
    export_decision_log_artifacts,
    validate_decision_log,
)


def main():
    print("==================================================")
    print("STAGE 16: Building Golden Set Decision Log Artifacts")
    print("==================================================")

    log_data = build_golden_decision_log()
    errors = validate_decision_log(log_data)

    if errors:
        print("[ERROR] Decision log integrity check failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    json_path, md_path = export_decision_log_artifacts(log_data)
    s = log_data["summary"]

    print("✓ Decision Log Built & Validated Successfully!")
    print(f"  - Total Records:      {s['total_examples']}")
    print(f"  - Verified:           {s['verified_examples']} (100.0%)")
    print(f"  - Confirmed:          {s['confirmed_count']} ({s['confirmed_rate']:.1%})")
    print(f"  - Changed/Corrected:  {s['changed_count']} ({s['changed_rate']:.1%})")
    print(f"  - JSON Artifact:      {json_path.relative_to(REPO_ROOT)}")
    print(f"  - Markdown Artifact:  {md_path.relative_to(REPO_ROOT)}")
    print("==================================================")


if __name__ == "__main__":
    main()
