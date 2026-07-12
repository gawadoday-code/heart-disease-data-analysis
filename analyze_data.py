"""Run compact EDA and statistical tests, then save reproducible outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, spearmanr

DATA = Path("data/processed/heart_disease_analysis_ready.csv")
OUTPUT = Path("reports/analysis_summary.json")
TABLE = Path("reports/key_findings.csv")

NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope"]

def rank_biserial(u: float, n1: int, n2: int) -> float:
    return 1 - (2 * u) / (n1 * n2)

def cramers_v(table: pd.DataFrame) -> float:
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    denominator = n * min(r - 1, k - 1)
    return float(np.sqrt(chi2 / denominator)) if denominator else 0.0

def main() -> None:
    df = pd.read_csv(DATA)
    findings = []

    for col in NUMERIC:
        no = df.loc[df["disease_present"].eq("No"), col].dropna()
        yes = df.loc[df["disease_present"].eq("Yes"), col].dropna()
        test = mannwhitneyu(yes, no, alternative="two-sided")
        findings.append({
            "question": f"{col} vs disease",
            "test": "Mann-Whitney U",
            "valid_n": int(len(no) + len(yes)),
            "group_no_median": float(no.median()),
            "group_yes_median": float(yes.median()),
            "p_value": float(test.pvalue),
            "effect_size": float(rank_biserial(test.statistic, len(yes), len(no))),
        })

    for col in CATEGORICAL:
        table = pd.crosstab(df[col], df["disease_present"])
        chi2, p, _, expected = chi2_contingency(table)
        findings.append({
            "question": f"{col} vs disease",
            "test": "Chi-square",
            "valid_n": int(table.to_numpy().sum()),
            "p_value": float(p),
            "effect_size": cramers_v(table),
            "minimum_expected_count": float(expected.min()),
        })

    pair = df[["age", "thalch"]].dropna()
    rho, p = spearmanr(pair["age"], pair["thalch"])
    findings.append({
        "question": "age vs thalch",
        "test": "Spearman correlation",
        "valid_n": int(len(pair)),
        "p_value": float(p),
        "effect_size": float(rho),
    })

    summary = {
        "rows": int(len(df)),
        "disease_prevalence": float(df["disease_present"].eq("Yes").mean()),
        "male_share": float(df["sex"].eq("Male").mean()),
        "missingness_by_column": df.isna().sum().to_dict(),
        "findings": findings,
        "causal_claim_allowed": False,
    }
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(findings).to_csv(TABLE, index=False)
    print(f"Saved {OUTPUT} and {TABLE}")

if __name__ == "__main__":
    main()
