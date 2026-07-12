"""Validate repository completeness and README local links."""
from pathlib import Path
import re

ROOT = Path(".")
README = ROOT / "README.md"
required = [
    "assets/dashboard_preview.png",
    "dashboard/01_executive_overview.png",
    "dashboard/02_clinical_insights.png",
    "dashboard/03_quality_and_model.png",
    "dashboard/heart_disease_dashboard_3_pages.pdf",
    "data/raw/heart_disease_uci.csv",
    "data/processed/heart_disease_analysis_ready.csv",
    "docs/ANALYTICAL_DECISIONS.md",
    "docs/DATA_DICTIONARY.md",
    "docs/INTERVIEW_NOTES.md",
    "docs/MODEL_CARD.md",
    "src/clean_data.py",
    "src/analyze_data.py",
    "src/train_model.py",
    "run_pipeline.py",
    "requirements.txt",
]
missing = [path for path in required if not (ROOT / path).exists()]
if missing:
    raise FileNotFoundError(f"Missing required files: {missing}")

content = README.read_text(encoding="utf-8")
links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
broken = []
for link in links:
    if link.startswith(("http://", "https://", "#", "<")):
        continue
    if not (ROOT / link).exists():
        broken.append(link)
if broken:
    raise FileNotFoundError(f"Broken README links: {broken}")

print("Repository validation passed.")
