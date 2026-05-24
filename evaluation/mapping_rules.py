# mapping_rules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

def _find_contains(tasks, phrase):
    p = _norm(phrase)
    return [i for i, t in enumerate(tasks) if p in _norm(t)]


def _norm(s: str) -> str:
    return " ".join(str(s).strip().upper().split())


@dataclass(frozen=True)
class MappingResult:
    groups: Dict[str, List[int]]
    used_task_indices: List[int]


# helpers
def _build_exact_index(tasks: Sequence[str]) -> Dict[str, List[int]]:
    d: Dict[str, List[int]] = {}
    for i, t in enumerate(tasks):
        d.setdefault(_norm(t), []).append(i)
    return d


def _find_exact(exact_index: Dict[str, List[int]], stmt: str) -> List[int]:
    return exact_index.get(_norm(stmt), [])


def _find_startswith(tasks: Sequence[str], prefix: str) -> List[int]:
    p = _norm(prefix)
    out = []
    for i, t in enumerate(tasks):
        if _norm(t).startswith(p):
            out.append(i)
    return out


def build_groups_from_tasks(tasks150: Sequence[str], weights_classes27: Sequence[str]) -> MappingResult:
    tasks = list(tasks150)
    exact_index = _build_exact_index(tasks)

    # mapping
    # Keys MUST be exactly those from weights.csv.
    whitelist: Dict[str, List[str]] = {
        # AF / AFL
        "164889003": ["ATRIAL FIBRILLATION"],
        "164890007": ["ATRIAL FLUTTER"],

        # BBB generic: leave empty on purpose (task list doesn't contain "BUNDLE BRANCH BLOCK")
        "6374002": [],

        # Bradycardia generic (426627000): not present as exact in tasks -> leave empty
        "426627000": [],

        # LBBB group in weights.csv
        "733534002|164909002": ["LEFT BUNDLE BRANCH BLOCK"],
        
        # RBBB group in weights.csv
        "713427006|59118001": ["RIGHT BUNDLE BRANCH BLOCK"],

        # 1st degree AV block
        "270492004": ["WITH 1ST DEGREE AV BLOCK"],


        # IRBBB
        "713426002": ["INCOMPLETE RIGHT BUNDLE BRANCH BLOCK"],

        # Axis / fascicular
        "39732003": ["LEFT AXIS DEVIATION"],

        "445118002": ["LEFT ANTERIOR FASCICULAR BLOCK"],

        "47665007": ["RIGHT AXIS DEVIATION"],

        # Prolonged PR interval (164947007): no exact statement in tasks -> leave empty
        "164947007": [],

        # Low voltage QRS
        "251146004": ["LOW VOLTAGE QRS"],

        # Prolonged QT
        "111975006": ["PROLONGED QT"],

        # NSIVCD
        "698252002": ["NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY"],


        # Sinus rhythm
        "426783006": ["SINUS RHYTHM"],

        # PAC/SVPB group in weights.csv
        "284470004|63593006": ["PREMATURE ATRIAL COMPLEXES", "PREMATURE SUPRAVENTRICULAR COMPLEXES"],
        
    
        # Pacing rhythm
        "10370003": ["VENTRICULAR-PACED RHYTHM", "ATRIAL-PACED RHYTHM"],

        # PRWP (365413008): not in tasks -> leave empty
        "365413008": [],

        # PVC/VPB group in weights.csv
        "427172004|17338001": ["PREMATURE VENTRICULAR COMPLEXES"],

        # QAb (164917005): not in tasks -> leave empty
        "164917005": [],

        # Sinus arrhythmia
        "427393009": ["WITH SINUS ARRHYTHMIA"],

        # Sinus bradycardia
        "426177001": ["SINUS BRADYCARDIA"],

        # Sinus tachycardia
        "427084000": ["SINUS TACHYCARDIA"],

        # T wave abnormality
        "164934002": ["NONSPECIFIC T WAVE ABNORMALITY"],

        # T wave inversion
        "59931005": ["T WAVE INVERSION LESS EVIDENT IN", "T WAVE INVERSION MORE EVIDENT IN", "T WAVE INVERSION NOW EVIDENT IN"],
    }

    groups: Dict[str, List[int]] = {c: [] for c in weights_classes27}

    
    for cls in weights_classes27:
        idxs: List[int] = []
        for stmt in whitelist.get(cls, []):
            idxs.extend(_find_exact(exact_index, stmt))
        groups[cls] = sorted(set(idxs))



    used = sorted({i for lst in groups.values() for i in lst})
    return MappingResult(groups=groups, used_task_indices=used)
