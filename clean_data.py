"""Clean the raw dataset and write an auditable processed copy."""
from pathlib import Path
import json
import pandas as pd

RAW = Path("data/raw/heart_disease_uci.csv")
OUTPUT = Path("data/processed/heart_disease_analysis_ready.csv")
AUDIT = Path("reports/cleaning_summary.json")

EXPECTED = [
    "id", "age", "sex", "dataset", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

def main() -> None:
    df = pd.read_csv(RAW)
    if set(df.columns) != set(EXPECTED):
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

    before_rows = len(df)
    before_missing = df.isna().sum().to_dict()

    df["chol_original"] = df["chol"]
    df["trestbps_original"] = df["trestbps"]
    df["thal_original"] = df["thal"]
    df["thal"] = df["thal"].replace({"reversable defect": "reversible defect"})

    for col in ["fbs", "exang"]:
        df[col] = df[col].replace({
            True: "Yes", False: "No", "TRUE": "Yes", "FALSE": "No",
            "True": "Yes", "False": "No"
        })

    chol_zero_count = int(df["chol"].eq(0).sum())
    bp_zero_count = int(df["trestbps"].eq(0).sum())
    df["chol_zero_reclassified"] = df["chol"].eq(0)
    df["trestbps_zero_reclassified"] = df["trestbps"].eq(0)
    df.loc[df["chol"].eq(0), "chol"] = pd.NA
    df.loc[df["trestbps"].eq(0), "trestbps"] = pd.NA

    df["disease_present"] = df["num"].gt(0).map({True: "Yes", False: "No"})
    duplicate_subset = [c for c in EXPECTED if c != "id"]
    df["potential_duplicate"] = df.duplicated(subset=duplicate_subset, keep=False)

    output_columns = EXPECTED + [
        "disease_present", "chol_original", "trestbps_original", "thal_original",
        "chol_zero_reclassified", "trestbps_zero_reclassified", "potential_duplicate"
    ]
    df[output_columns].to_csv(OUTPUT, index=False)

    summary = {
        "rows_before": before_rows,
        "rows_after": len(df),
        "rows_deleted": before_rows - len(df),
        "missing_before": before_missing,
        "missing_after": df[EXPECTED].isna().sum().to_dict(),
        "chol_zero_reclassified": chol_zero_count,
        "trestbps_zero_reclassified": bp_zero_count,
        "potential_duplicate_rows": int(df["potential_duplicate"].sum()),
    }
    AUDIT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
