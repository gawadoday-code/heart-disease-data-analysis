"""Run the complete reproducible workflow."""
import subprocess
import sys

for script in ["src/clean_data.py", "src/analyze_data.py", "src/train_model.py"]:
    print(f"\n>>> Running {script}")
    subprocess.run([sys.executable, script], check=True)

print("\nWorkflow completed successfully.")
