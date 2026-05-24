import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REALISTIC_ARTIFACTS = {
    "ma": [7, 3, -2],
    "em": [4, 1, -3],
}

SIMULATED_ARTIFACTS = {
    "gn": [1, -4, -7],
    "dn": [9, 4, -6],
    "in": [0.007, 0.018, 0.04],
}


def run_generator(script_path: Path, artifact: str, severity_name: str, severity_value: float):
    print("=" * 80)
    print(f"Running {script_path.name}")
    print(f"Artifact: {artifact}")
    print(f"Severity: {severity_name}")
    print(f"Value: {severity_value}")
    print("=" * 80)

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--artifact",
            artifact,
            "--severity-name",
            severity_name,
            "--severity-value",
            str(severity_value),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

def run_verification(script_path: Path, artifact: str, severity_name: str, severity_value: float):
    print("=" * 80)
    print(f"Verifying {artifact} {severity_name}")
    print(f"Value: {severity_value}")
    print("=" * 80)

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--artifact",
            artifact,
            "--severity-name",
            severity_name,
            "--severity-value",
            str(severity_value),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

def main():
    realistic_script = PROJECT_ROOT / "benchmark_generation" / "bm_generator.py"
    simulated_script = PROJECT_ROOT / "benchmark_generation" / "bm_generator_simulated.py"

    verification_script = PROJECT_ROOT / "benchmark_generation" / "bm_verification.py"
    impulse_verification_script = PROJECT_ROOT / "benchmark_generation" / "bm_inverification.py"

    for artifact, values in REALISTIC_ARTIFACTS.items():
        for i, value in enumerate(values, start=1):
            severity_name = f"Sev{i}"

            run_generator(
                script_path=realistic_script,
                artifact=artifact,
                severity_name=severity_name,
                severity_value=value,
            )

            run_verification(
                script_path=verification_script,
                artifact=artifact,
                severity_name=severity_name,
                severity_value=value,
            )

    for artifact, values in SIMULATED_ARTIFACTS.items():
        for i, value in enumerate(values, start=1):
            severity_name = f"Sev{i}"

            run_generator(
                script_path=simulated_script,
                artifact=artifact,
                severity_name=severity_name,
                severity_value=value,
            )

            if artifact == "in":
                verifier = impulse_verification_script
            else:
                verifier = verification_script

            run_verification(
                script_path=verifier,
                artifact=artifact,
                severity_name=severity_name,
                severity_value=value,
            )


if __name__ == "__main__":
    main()