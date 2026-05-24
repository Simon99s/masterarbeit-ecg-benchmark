# config.py
from pathlib import Path

# directory where this file lives
PROJECT_ROOT = Path(__file__).resolve().parent

PHYSIONET_ROOT = (
    PROJECT_ROOT
    / "physionet.org"
    / "files"
    / "challenge-2021"
    / "1.0.3"
    / "training"
)
NSTDB_ROOT = PROJECT_ROOT / "mit-bih-noise-stress-test-database-1.0.0"

FS = 500

DEBUG_N_RECORDS = 500   # ← only test on 50 records

ARTIFACTS = [
    "em",   # electrode motion
    "ma",   # muscle artifact
    "gn",   # gaussian noise
    "dn",   # discretization
    "in",   # impulse noise
]

ARTIFACT_LEVELS = {
    "em": [4, -1, -4],
    "ma": [4, 1, -3],
    "gn": [-1, -5, -7],
    "dn": [9, 5, -2],
    "in": [
        {"p": 0.01, "amp": 10},
        {"p": 0.021, "amp": 10},
        {"p": 0.044, "amp": 10},
    ],
}