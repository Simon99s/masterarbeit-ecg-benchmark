import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_COMMANDS = [
    [
        PROJECT_ROOT / "training" / "xecg" / "train_xecg_emory21_finetune.py",
        "--config",
        PROJECT_ROOT / "training" / "xecg" / "config_train_xecg_emory21.yaml",
    ],
    [
        PROJECT_ROOT / "training" / "st_mem" / "train_stmem_emory21.py",
        "--config",
        PROJECT_ROOT / "training" / "st_mem" / "config_train_stmem_emory21.yaml",
    ],
    [
        PROJECT_ROOT / "training" / "jepa" / "train_jepa.py",
        "--config",
        PROJECT_ROOT / "training" / "jepa" / "config_train_jepa.yaml",
    ],
]


def run_training(command):
    script_path = command[0]

    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    for arg in command[1:]:
        if isinstance(arg, Path) and arg.suffix in [".yaml", ".yml"] and not arg.exists():
            raise FileNotFoundError(f"Config file not found: {arg}")

    print("=" * 80)
    print(f"Running training script: {script_path.relative_to(PROJECT_ROOT)}")
    print("=" * 80)

    subprocess.run(
        [sys.executable] + [str(x) for x in command],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    for command in TRAINING_COMMANDS:
        run_training(command)


if __name__ == "__main__":
    main()