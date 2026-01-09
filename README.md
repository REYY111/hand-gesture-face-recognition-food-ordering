Sistem ini memungkinkan pengguna memesan makanan hanya dengan **mengenali wajah** dan **gerakan tangan** untuk mendeteksi jenis serta jumlah makanan. Dilengkapi fitur pengelolaan saldo, pelatihan AI, dan enkripsi data untuk keamanan.

---

## 🚀 Petunjuk Penggunaan

Berikut fungsi dari masing-masing file Python:

| File Python             | Deskripsi                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| `ambildatamuka.py`      | Mengambil dan menyimpan data wajah pengguna.                            |
| `ambildatatangan.py`    | Mengambil data gestur tangan (jenis & jumlah makanan).                  |
| `convertdata.py`        | Mengonversi data dari format `.pkl` ke `.csv`.                          |
| `cekdatacsv.py`         | (Opsional) Mengecek isi dan struktur file `.csv`.                       |
| `trainingdata.py`       | Melatih model AI (Random Forest & Neural Network) dengan data pengguna. |
| `tambahsaldo.py`        | Menambah saldo pengguna & mengatur harga makanan.                       |
| `full.py`               | Menjalankan sistem pemesanan secara penuh (face & gesture detection).   |
| `enkripsi.py`           | Mengenkripsi file `.csv` menggunakan metode Fernet.                     |
| `bersihkan_pesanan.py`  | Membersihkan file data pesanan dari karakter/simbol aneh.               |

---

## 🔁 Alur Jalannya Program

1. 📸 Jalankan `ambildatamuka.py` untuk data wajah.
2. ✋ Jalankan `ambildatatangan.py` untuk data gestur tangan.
3. 💰 Gunakan `tambahsaldo.py` untuk input saldo dan harga.
4. 🔄 Jalankan `convertdata.py` untuk ubah ke format CSV.
5. 🧠 Latih model dengan `trainingdata.py`.
6. 🎯 Jalankan sistem utama dengan `full.py`.

> Opsional:
> - Cek data CSV: `cekdatacsv.py`
> - Enkripsi: `enkripsi.py`
> - Bersih-bersih data pesanan: `bersihkan_pesanan.py`

---

## 👨‍💻 Author

**Reyhan Andhata Pratama**  
NIM: `1102223153`  
Telkom University – S1 Teknik Elektro  

**Michael Pardede**  
NIM: `1102223136`  
Telkom University – S1 Teknik Elektro  
