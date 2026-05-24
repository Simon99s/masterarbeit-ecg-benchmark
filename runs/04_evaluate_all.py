import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_SCRIPT = PROJECT_ROOT / "evaluation" / "evaluation.py"

MODELS = [
    "ECGFounder",
    "jepa",
    "st-mem",
    "xecg",
]


def run_evaluation(model_name: str):
    if not EVALUATION_SCRIPT.exists():
        raise FileNotFoundError(f"Evaluation script not found: {EVALUATION_SCRIPT}")

    print("=" * 80)
    print(f"Evaluating model: {model_name}")
    print("=" * 80)

    subprocess.run(
        [
            sys.executable,
            str(EVALUATION_SCRIPT),
            "--model",
            model_name,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    for model_name in MODELS:
        run_evaluation(model_name)


if __name__ == "__main__":
    main()