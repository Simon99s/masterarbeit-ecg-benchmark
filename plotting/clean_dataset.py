from pathlib import Path
import wfdb
import numpy as np
import pandas as pd

class PhysioNetClean:

    def __init__(self, root, label_csv, limit=None):

        self.root = Path(root)

        # load labels
        self.labels_df = pd.read_csv(label_csv)

        # build lookup: relative_path_without_extension -> label
        self.label_dict = {}

        for _, row in self.labels_df.iterrows():

            rel_path = Path(row["mat_relpath"]).with_suffix("")
            rel_path_str = str(rel_path).replace("\\", "/")

            label = np.array(eval(row["label"]), dtype=np.float32)

            self.label_dict[rel_path_str] = label

        # build list of records that have labels
        self.records = []

        for hea_path in sorted(self.root.rglob("*.hea")):

            rel_path = hea_path.relative_to(self.root).with_suffix("")
            rel_path_str = str(rel_path).replace("\\", "/")

            if rel_path_str in self.label_dict:
                self.records.append(hea_path)

        if limit is not None:
            self.records = self.records[:limit]

    def __len__(self):
        return len(self.records)

    def load_record(self, idx):

        hea_path = self.records[idx]

        rel_path = hea_path.relative_to(self.root).with_suffix("")
        rel_path_str = str(rel_path).replace("\\", "/")

        record_path = hea_path.with_suffix("")
        record_id = record_path.name

        sig, _ = wfdb.rdsamp(str(record_path))
        clean = np.nan_to_num(sig.T.astype(np.float32), nan=0.0)

        label = self.label_dict[rel_path_str]

        return clean, label, record_id