import pickle
import pandas as pd

with open("faces_data.pickle", "rb") as f:
    faces_data, labels = pickle.load(f)

# Flatten setiap face 50x50 jadi 2500 fitur
flattened_faces = [face.flatten() for face in faces_data]

# Buat DataFrame
df = pd.DataFrame(flattened_faces)
df.insert(0, 'label', labels)

# Simpan ke CSV
df.to_csv("data_wajah.csv", index=False)

print("Data pickle berhasil diubah menjadi CSV untuk training ML.")
