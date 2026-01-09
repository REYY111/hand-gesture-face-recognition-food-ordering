import pandas as pd

df = pd.read_csv("data_wajah.csv", header=0)  # pastikan header=0
print(df.head())  # cek 5 baris pertama
print(df.dtypes)  # cek tipe data tiap kolom

# Lihat apakah ada baris yang isi 'x0' di kolom fitur
print(df[df['x0'] == 'x0'])  # harus kosong jika header benar
