import pandas as pd

w = pd.read_csv("weights_21.csv", index_col=0)

print("shape:", w.shape)
print("rows == cols:", list(w.index) == list(w.columns))
print("diag:", w.values.diagonal())
print("all diag == 1:", (w.values.diagonal() == 1).all())
print("symmetric:", (w.values == w.values.T).all())
