import pickle
import os

FACES_FILE = "faces_data.pickle"
USER_FILE = "users_data.pickle"
HARGA_FILE = "harga.txt"

# ANSI Color
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def cls():
    os.system("cls" if os.name == "nt" else "clear")

def pause():
    if os.name == "nt":
        os.system("pause")
    else:
        input("Tekan Enter untuk melanjutkan...")

# ------------------------ User / Saldo ------------------------ #

def load_users_from_faces():
    if not os.path.exists(FACES_FILE):
        print(f"{RED}File faces_data.pickle tidak ditemukan.{RESET}")
        return {}

    with open(FACES_FILE, "rb") as f:
        _, labels = pickle.load(f)

    unique_users = set(labels)
    users = {}

    for name in unique_users:
        users[name] = {"saldo": 0}
    return users

def load_user_data():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return {}

def save_user_data(users):
    with open(USER_FILE, "wb") as f:
        pickle.dump(users, f)

def tampilkan_users(users):
    print(f"\n{CYAN}Daftar User:{RESET}")
    for name, data in users.items():
        print(f"  - {YELLOW}{name}{RESET}: Saldo = {GREEN}Rp{data['saldo']}{RESET}")

def tambah_saldo(users):
    username = input("Masukkan nama user: ")
    if username not in users:
        print(f"{RED}User tidak ditemukan.{RESET}")
        return
    try:
        jumlah = int(input("Jumlah saldo yang ditambahkan: "))
        users[username]["saldo"] += jumlah
        print(f"{GREEN}Saldo baru {username}: Rp{users[username]['saldo']}{RESET}")
    except ValueError:
        print(f"{RED}Input harus angka.{RESET}")

def edit_saldo(users):
    username = input("Masukkan nama user: ")
    if username not in users:
        print(f"{RED}User tidak ditemukan.{RESET}")
        return
    try:
        saldo_baru = int(input("Masukkan saldo baru: "))
        users[username]["saldo"] = saldo_baru
        print(f"{YELLOW}Saldo {username} diubah menjadi: Rp{saldo_baru}{RESET}")
    except ValueError:
        print(f"{RED}Input harus angka.{RESET}")

# ------------------------ Harga / Menu ------------------------ #

def load_harga():
    harga = {}
    if os.path.exists(HARGA_FILE):
        with open(HARGA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # lewati baris kosong
                if ":" in line:
                    try:
                        item, hrg = line.split(":")
                        harga[item.strip().upper()] = int(hrg.strip())
                    except ValueError:
                        print(f"{RED}Format salah: {line}{RESET}")
                elif " " in line:
                    try:
                        item, hrg = line.split()
                        harga[item.strip().upper()] = int(hrg.strip())
                    except ValueError:
                        print(f"{RED}Format salah: {line}{RESET}")
                else:
                    print(f"{YELLOW}Lewati baris tidak valid: {line}{RESET}")
    return harga


def simpan_harga(harga_dict):
    with open(HARGA_FILE, "w", encoding="utf-8") as f:
        for item, hrg in harga_dict.items():
            f.write(f"{item}: {hrg}\n")  # gunakan titik dua untuk pemisah



def tampilkan_harga(harga_dict):
    print(f"\n{CYAN}Daftar Harga Makanan:{RESET}")
    for item, hrg in harga_dict.items():
        print(f"  - {YELLOW}{item.title()}{RESET}: Rp{GREEN}{hrg}{RESET}")

def tambah_harga(harga_dict):
    item = input("Nama makanan: ").upper()
    if item in harga_dict:
        print(f"{RED}Item sudah ada.{RESET}")
        return
    try:
        hrg = int(input("Harga (Rp): "))
        harga_dict[item] = hrg
        print(f"{GREEN}{item.title()} ditambahkan.{RESET}")
    except ValueError:
        print(f"{RED}Input harus angka.{RESET}")

def edit_harga(harga_dict):
    item = input("Nama makanan yang ingin diubah: ").upper()
    if item not in harga_dict:
        print(f"{RED}Item tidak ditemukan.{RESET}")
        return
    try:
        hrg = int(input("Harga baru (Rp): "))
        harga_dict[item] = hrg
        print(f"{YELLOW}Harga {item.title()} diubah menjadi Rp{hrg}{RESET}")
    except ValueError:
        print(f"{RED}Input harus angka.{RESET}")

# ------------------------ Menu Utama ------------------------ #

def main():
    base_users = load_users_from_faces()
    current_users = load_user_data()

    for name in base_users:
        if name not in current_users:
            current_users[name] = {"saldo": 0}

    harga = load_harga()

    while True:
        cls()
        print(f"\n{CYAN}=== MENU UTAMA ==={RESET}")
        print(f"{YELLOW}1.{RESET} Tampilkan User & Saldo")
        print(f"{YELLOW}2.{RESET} Tambah Saldo")
        print(f"{YELLOW}3.{RESET} Edit Saldo")
        print(f"{YELLOW}4.{RESET} Tampilkan Harga Makanan")
        print(f"{YELLOW}5.{RESET} Tambah Harga Makanan")
        print(f"{YELLOW}6.{RESET} Edit Harga Makanan")
        print(f"{YELLOW}7.{RESET} Simpan & Keluar")

        pilihan = input("Pilih opsi (1-7): ")

        if pilihan == '1':
            tampilkan_users(current_users)
        elif pilihan == '2':
            tambah_saldo(current_users)
        elif pilihan == '3':
            edit_saldo(current_users)
        elif pilihan == '4':
            tampilkan_harga(harga)
        elif pilihan == '5':
            tambah_harga(harga)
        elif pilihan == '6':
            edit_harga(harga)
        elif pilihan == '7':
            save_user_data(current_users)
            simpan_harga(harga)
            print(f"{GREEN}Data disimpan. Keluar dari program.{RESET}")
            break
        else:
            print(f"{RED}Pilihan tidak valid.{RESET}")

        pause()

if __name__ == "__main__":
    main()
