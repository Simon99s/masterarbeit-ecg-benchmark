import csv

INPUT = "weights.csv"
OUTPUT = "weights_21.csv"

WHITELIST_21 = [
    "164889003",
    "164890007",
    "733534002|164909002",
    "713427006|59118001",
    "270492004",
    "713426002",
    "39732003",
    "445118002",
    "47665007",
    "251146004",
    "111975006",
    "698252002",
    "426783006",
    "284470004|63593006",
    "10370003",
    "427172004|17338001",
    "427393009",
    "426177001",
    "427084000",
    "164934002",
    "59931005",
]

# ------------------------
# Load original CSV
# ------------------------
with open(INPUT, newline="") as f:
    reader = list(csv.reader(f))

header = reader[0][1:]     # column class labels
rows = reader[1:]          # [row_class, values...]

# Map column class → index
class_to_col = {c: i for i, c in enumerate(header)}

# Map row class → full row
row_map = {row[0]: row[1:] for row in rows}

# Sanity check
missing_cols = [c for c in WHITELIST_21 if c not in class_to_col]
missing_rows = [c for c in WHITELIST_21 if c not in row_map]
if missing_cols or missing_rows:
    raise ValueError(
        f"Missing classes in weights.csv:\n"
        f"columns: {missing_cols}\nrows: {missing_rows}"
    )

keep_col_idx = [class_to_col[c] for c in WHITELIST_21]

# ------------------------
# Write reduced CSV
# ------------------------
with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)

    # Header (EMPTY cell + class names)
    writer.writerow([""] + WHITELIST_21)

    # Rows in EXACT SAME ORDER as columns
    for cls in WHITELIST_21:
        full_row = row_map[cls]
        reduced = [full_row[i] for i in keep_col_idx]
        writer.writerow([cls] + reduced)

print("✅ Written:", OUTPUT)
