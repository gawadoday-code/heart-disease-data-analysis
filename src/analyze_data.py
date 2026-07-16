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
    """Positive values mean the first group tends to have larger values."""
    return (2 * u) / (n1 * n2) - 1


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rank_index in range(m - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        candidate = float(p_values[original_index]) * m / rank
        running = min(running, candidate)
        adjusted[original_index] = min(running, 1.0)
    return adjusted.tolist()

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

    adjusted = benjamini_hochberg([row["p_value"] for row in findings])
    for row, fdr_p_value in zip(findings, adjusted):
        row["fdr_p_value"] = fdr_p_value

    summary = {
        "rows": int(len(df)),
        "disease_prevalence": float(df["disease_present"].eq("Yes").mean()),
        "male_share": float(df["sex"].eq("Male").mean()),
        "missingness_by_column": df.isna().sum().to_dict(),
        "findings": findings,
        "causal_claim_allowed": False,
        "multiple_testing": "Benjamini-Hochberg FDR across reported tests",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(findings).to_csv(TABLE, index=False)
    print(f"Saved {OUTPUT} and {TABLE}")

if __name__ == "__main__":
    main()
