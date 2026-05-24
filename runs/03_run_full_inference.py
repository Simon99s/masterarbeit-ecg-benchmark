import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INFERENCE_SCRIPTS = [
    PROJECT_ROOT / "inference" / "ECGFounder" / "ECGFounder_inference.py",
    PROJECT_ROOT / "inference" / "jepa" / "jepa_inference.py",
    PROJECT_ROOT / "inference" / "st-mem" / "stmem_inference.py",
    PROJECT_ROOT / "inference" / "xecg" / "xecg_inference.py",
]


def run_inference(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Inference script not found: {script_path}")

    print("=" * 80)
    print(f"Running inference script: {script_path.relative_to(PROJECT_ROOT)}")
    print("=" * 80)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    for script_path in INFERENCE_SCRIPTS:
        run_inference(script_path)


if __name__ == "__main__":
    main()