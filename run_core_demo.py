from __future__ import annotations

from pathlib import Path

from config import DEFAULT_PATIENT_FILE
from core.export import export_excel
from core.pipeline import run_analysis


def main() -> None:
    result = run_analysis(DEFAULT_PATIENT_FILE, mode="mock_rules")
    print(result["summary"].to_string(index=False))
    output = export_excel(result["summary"], result["details"], result.get("similarity_table"))
    print(f"\nExported: {output}")


if __name__ == "__main__":
    main()

