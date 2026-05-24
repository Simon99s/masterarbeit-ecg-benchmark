import hashlib
import numpy as np
from artifact_engine import apply_artifact

class ECGArtifactBenchmark:

    def __init__(self, base_dataset, artifact_type, severity, fs=500):

        self.base = base_dataset
        self.artifact_type = artifact_type
        self.severity = severity
        self.fs = fs

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):

        clean, label, record_id = self.base.load_record(idx)

        # deterministic per record
        seed = int(hashlib.sha1(record_id.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)

        corrupted = apply_artifact(
            ecg=clean,
            artifact_type=self.artifact_type,
            severity=self.severity,
            fs=self.fs,
            rng=rng
        )

        return corrupted, label, record_id