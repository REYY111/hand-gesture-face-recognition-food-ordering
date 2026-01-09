from cryptography.fernet import Fernet

# 1. Generate Key dan simpan (jika belum ada)
key = Fernet.generate_key()
with open("key.key", "wb") as key_file:
    key_file.write(key)

# 2. Load file CSV dan baca sebagai biner
with open("data_wajah.csv", "rb") as file:
    data = file.read()

# 3. Enkripsi data
cipher = Fernet(key)
encrypted_data = cipher.encrypt(data)

# 4. Simpan hasil enkripsi ke file baru
with open("data_wajah_enkripsi.csv", "wb") as encrypted_file:
    encrypted_file.write(encrypted_data)

print("✅ File 'data_wajah.csv' berhasil dienkripsi dan disimpan sebagai 'data_wajah_enkripsi.csv'")
