import h5py

H5_PATH = r"C:\Users\simon\Desktop\heedb_i0006_100.h5"
with h5py.File(H5_PATH, "r") as f:
    print("label attrs:", list(f["label"].attrs.keys()))
    for k, v in f["label"].attrs.items():
        print(k, type(v), v if isinstance(v, (str, int, float)) else (len(v), str(v)[:80]))