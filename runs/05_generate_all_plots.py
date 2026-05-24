import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLOT_SCRIPTS = [
    # Add your actual plotting script filenames here
    PROJECT_ROOT / "plotting" / "analysis.py",
    PROJECT_ROOT / "plotting" / "analyze_cinc.py",
    PROJECT_ROOT / "plotting" / "classcontribution.py",
    PROJECT_ROOT / "plotting" / "confidence.py",
    PROJECT_ROOT / "plotting" / "corruption_examples.py",
    PROJECT_ROOT / "plotting" / "plot_bm_examples.py",
    PROJECT_ROOT / "plotting" / "plot_lead5.py",
    PROJECT_ROOT / "plotting" / "redundancy_inequity.py",
    PROJECT_ROOT / "plotting" / "plot_redundancy_inequity.py",
    PROJECT_ROOT / "plotting" / "precisionrecall.py",
    PROJECT_ROOT / "plotting" / "precisionrecallanalysis.py",
    PROJECT_ROOT / "plotting" / "SR_analysis.py",
]


def run_plot_script(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Plot script not found: {script_path}")

    print("=" * 80)
    print(f"Running plot script: {script_path.relative_to(PROJECT_ROOT)}")
    print("=" * 80)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    for script_path in PLOT_SCRIPTS:
        run_plot_script(script_path)


if __name__ == "__main__":
    main()