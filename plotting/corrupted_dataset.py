import hashlib
import numpy as np
from artifact_engine import apply_artifact

class ECGArtifactBenchmark:

    def __init__(self, clean_dataset, artifact_type, severity, fs=500):
        self.clean = clean_dataset
        self.artifact_type = artifact_type
        self.severity = severity
        self.fs = fs

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):

        clean, label, record_id = self.clean.load_record(idx)

        # deterministic per record
        seed = int(hashlib.sha1(record_id.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)

        corrupted_full = apply_artifact(
            ecg=clean,
            artifact_type=self.artifact_type,
            severity=self.severity,
            fs=self.fs,
            rng=rng
        )

        # -------------------------
        # 🔥 DEBUG CORRUPTION STRENGTH
        # -------------------------
        if idx < 5 and self.artifact_type == "in":
            diff = np.mean(np.abs(corrupted_full - clean))
            print(f"[DEBUG] Severity={self.severity} | Mean |corruption|={diff:.6f}")

        return corrupted_full, label, record_id