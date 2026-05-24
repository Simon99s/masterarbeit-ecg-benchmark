# utils.py

from pathlib import Path
import pandas as pd

RESULTS_FILE = Path("results.csv")

def save_result(artifact, severity, score):

    row = {
        "artifact": artifact,
        "severity": severity,
        "score": score
    }

    if RESULTS_FILE.exists():
        df = pd.read_csv(RESULTS_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(RESULTS_FILE, index=False)