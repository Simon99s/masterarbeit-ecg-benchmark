from pathlib import Path
from mapping_rules import build_groups_from_tasks
from models.ecgfounder.mapping_utils import (
    load_tasks,
    load_weights_classes,
    map_probs150_to_probs21,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TASKS_TXT = PROJECT_ROOT / "models" / "ecgfounder" / "tasks.txt"
WEIGHTS_CSV = PROJECT_ROOT / "models" / "ecgfounder" / "weights_21.csv"

tasks150 = load_tasks(TASKS_TXT)
weights_classes21 = load_weights_classes(WEIGHTS_CSV)

mapping = build_groups_from_tasks(tasks150, weights_classes21)


def postprocess_ecgfounder(probs150):
    return map_probs150_to_probs21(
        probs150,
        weights_classes21,
        mapping.groups
    )