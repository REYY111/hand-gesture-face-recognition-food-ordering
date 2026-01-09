import re

# File input/output
INPUT_FILE = "datapesanan.txt"
OUTPUT_FILE = "datapesanan_bersih.txt"
HARGA_FILE = "harga.txt"

# Fungsi untuk membersihkan karakter aneh
def bersihkan_teks(teks):
    return re.sub(r'[^\x20-\x7E]', '', teks).strip()

# Load harga dari harga.txt
def load_harga():
    harga_dict = {}
    try:
        with open(HARGA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    nama, harga = line.strip().split(":")
                    harga_dict[nama.strip().lower()] = int(harga.strip())
    except FileNotFoundError:
        print("File harga.txt tidak ditemukan. Melanjutkan tanpa menghitung total harga.")
    return harga_dict

# Proses pembersihan dan perhitungan
def proses_file():
    harga_dict = load_harga()
    try:
        with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File {INPUT_FILE} tidak ditemukan.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for line in lines:
            clean_line = bersihkan_teks(line)
            out.write(clean_line + "\n")

            # Coba hitung harga (opsional)
            try:
                if ":" in clean_line:
                    bagian = clean_line.split(":")[1]
                    item_list = bagian.split(",")
                    total = 0
                    for item in item_list:
                        parts = item.strip().split()
                        if len(parts) >= 2:
                            jumlah = int(parts[-2]) if parts[-2].isdigit() else int(parts[-1])
                            nama = " ".join(parts[:-2]) if parts[-2].isdigit() else " ".join(parts[:-1])
                            nama = bersihkan_teks(nama).lower()
                            harga = harga_dict.get(nama, 0)
                            total += harga * jumlah
                    if total > 0:
                        print(f"{clean_line} => Total: Rp{total}")
            except Exception as e:
                print(f"Gagal menghitung total dari baris: {clean_line} | Error: {e}")

if __name__ == "__main__":
    proses_file()
