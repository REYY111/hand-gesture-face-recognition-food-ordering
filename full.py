# ====== CONFIG SUPPRESS WARNING ======
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# ====== LIBRARY ======
import cv2
import numpy as np
import pickle
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import mediapipe as mp
import re
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 125)
engine.say("silahkan memesan")
engine.runAndWait()

# ====== LOAD FACE DATA ======
with open("faces_data.pickle", "rb") as f:
    faces_data, labels = pickle.load(f)

label_dict = {}
current_label = 0
numeric_labels = []

for label in labels:
    if label not in label_dict:
        label_dict[label] = current_label
        current_label += 1
    numeric_labels.append(label_dict[label])

faces_data = [np.array(face, dtype=np.uint8) for face in faces_data]
numeric_labels = np.array(numeric_labels)

# ====== LOAD USER SALDO DATA ======
USER_FILE = "users_data.pickle"
if os.path.exists(USER_FILE):
    with open(USER_FILE, "rb") as f:
        user_data = pickle.load(f)
else:
    user_data = {}

# ====== TRAIN FACE MODEL ======
face_model = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
face_model.train(faces_data, numeric_labels)

# ====== LOAD GESTURE MODELS ======
df_sibi = pd.read_csv('datamakanan.csv')
labels_sibi = df_sibi.iloc[:, 0]
features_sibi = df_sibi.iloc[:, 1:]
scaler_sibi = StandardScaler().fit(features_sibi)
model_sibi = RandomForestClassifier().fit(scaler_sibi.transform(features_sibi), labels_sibi)

df_kuantitas = pd.read_csv('kuantitas.csv')
labels_kuantitas = df_kuantitas.iloc[:, 0]
features_kuantitas = df_kuantitas.iloc[:, 1:]
scaler_kuantitas = StandardScaler().fit(features_kuantitas)
model_kuantitas = RandomForestClassifier().fit(scaler_kuantitas.transform(features_kuantitas), labels_kuantitas)

# ====== SETUP MEDIAPIPE ======
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ====== VARIABEL ======
mode = "Wajah"
recognized_name = None
saldo = 0
sibi_detected = False
kuantitas_detected = False
last_hand_lost_time = None
last_sibi = ""
last_kuantitas = ""
summary_start_time = None
pending_repeat_warning_time = None
data_saved = False
pesanan = []
pelanggan_count = 1
recognition_timer = None
countdown_start = None
last_recognized_name = None
required_duration = 4
prep_duration = 2
show_saldo_time = None

# ====== FUNGSI TAMBAHAN ======
def draw_fancy_text(img, text, position, font_scale=1, thickness=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    box_x2 = text_x + text_size[0] + 20
    box_y2 = text_y + 30
    overlay = img.copy()
    cv2.rectangle(overlay, (text_x - 10, text_y - 30), (box_x2, box_y2), (255, 192, 203), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

# ====== MAIN LOOP ======
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    current_time = time.time()

    if mode == "Wajah":
        name = "Unknown"
        for (x, y, fw, fh) in faces:
            face_roi = gray[y:y+fh, x:x+fw]
            face_resized = cv2.resize(face_roi, (50, 50))
            label_pred, confidence = face_model.predict(face_resized)
            name = next((n for n, l in label_dict.items() if l == label_pred), "Unknown")

            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
            draw_fancy_text(frame, f"{name} ({confidence:.2f})", (x, y-10))

            if name == last_recognized_name:
                if recognition_timer is None:
                    recognition_timer = current_time
                elapsed = current_time - recognition_timer
                if elapsed < prep_duration:
                    draw_fancy_text(frame, "Siapkan wajah Anda...", (10, 110))
                else:
                    if countdown_start is None:
                        countdown_start = current_time
                    countdown_elapsed = current_time - countdown_start
                    countdown_value = required_duration - int(countdown_elapsed)
                    if countdown_value > 0:
                        draw_fancy_text(frame, f"Konfirmasi dalam: {countdown_value}", (10, 110))
                    else:
                        recognized_name = name
                        saldo = user_data.get(name, {}).get('saldo', 0)
                        mode = "MAKANAN"
                        recognition_timer = None
                        countdown_start = None
                        last_recognized_name = None
            else:
                last_recognized_name = name
                recognition_timer = current_time
                countdown_start = None
            break

    elif mode in ["MAKANAN", "Kuantitas", "Selesai"]:
        draw_fancy_text(frame, f'{recognized_name} | Saldo: {saldo}', (10, 40))
        draw_fancy_text(frame, f'Masukan: {mode}', (10, 80))

        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            last_hand_lost_time = None
            for hand_landmarks in result.multi_hand_landmarks:
                coords = [c for lm in hand_landmarks.landmark for c in (lm.x, lm.y, lm.z)]
                if len(coords) == 63:
                    data_np = np.array(coords).reshape(1, -1)
                    if mode == "MAKANAN":
                        pred = model_sibi.predict(scaler_sibi.transform(data_np))[0]
                        last_sibi = pred
                        sibi_detected = True
                    elif mode == "Kuantitas":
                        pred = model_kuantitas.predict(scaler_kuantitas.transform(data_np))[0]
                        last_kuantitas = pred
                        kuantitas_detected = True
                    draw_fancy_text(frame, f'{mode} Gesture: {pred}', (10, 130))
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            if mode == "MAKANAN" and sibi_detected:
                if last_hand_lost_time is None:
                    last_hand_lost_time = current_time
                elif current_time - last_hand_lost_time > 2:
                    mode = "Kuantitas"
                    sibi_detected = False
            elif mode == "Kuantitas" and kuantitas_detected:
                if last_hand_lost_time is None:
                    last_hand_lost_time = current_time
                elif current_time - last_hand_lost_time > 2:
                    mode = "Selesai"
                    kuantitas_detected = False
                    summary_start_time = current_time
                    pending_repeat_warning_time = current_time

        if mode == "Selesai":
            center_x = w // 2 - 200
            center_y = h // 2
            draw_fancy_text(frame, f'Makanan: {last_sibi}', (center_x, center_y - 60), 1.2, 2)
            draw_fancy_text(frame, f'Kuantitas: {last_kuantitas} pcs', (center_x, center_y), 1.2, 2)
            if pending_repeat_warning_time and 3 <= current_time - pending_repeat_warning_time <= 5:
                cv2.putText(frame, "Ingin memesan lagi?", (center_x, center_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if result.multi_hand_landmarks and 3 <= current_time - pending_repeat_warning_time <= 5:
                pesanan.append(f"{last_sibi} {last_kuantitas} pcs")
                mode = "MAKANAN"
                last_sibi = ""
                last_kuantitas = ""
                summary_start_time = None
                pending_repeat_warning_time = None
            if summary_start_time and current_time - summary_start_time > 5:
                pesanan.append(f"{last_sibi} {last_kuantitas} pcs")

        if not data_saved and pesanan:
            with open("datapesanan.txt", "a") as data:
                data.write(f"Pelanggan {pelanggan_count:02d} ({recognized_name}): " + ", ".join(pesanan) + "\n")
            pelanggan_count += 1
            data_saved = True
            try:
                harga_dict = {}
                with open("harga.txt", "r") as f:
                    for line in f:
                        if ":" in line:
                            nama, harga = line.strip().split(":")
                            harga_dict[nama.strip().upper()] = int(harga.strip())
                total_harga_semua = 0
                for item in pesanan:
                    item_clean = item.replace("pcs", "").strip()
                    match = re.match(r'([^\d]+)\s+(\d+)', item_clean)
                    if match:
                        nama_makanan = re.sub(r'[^A-Z ]', '', match.group(1).upper()).strip()
                        jumlah = int(match.group(2))
                        harga = harga_dict.get(nama_makanan, 0)
                        total_harga_semua += harga * jumlah

                if recognized_name in user_data:
                    user_data[recognized_name]['saldo'] -= total_harga_semua
                    with open(USER_FILE, "wb") as f:
                        pickle.dump(user_data, f)
                    show_saldo_time = current_time
                    draw_fancy_text(frame, f"Saldo Anda berkurang {total_harga_semua}", (w//2 - 200, h//2 + 100), 1.2, 2)
                    print(f"Saldo {recognized_name} dikurangi {total_harga_semua}. Sisa saldo: {user_data[recognized_name]['saldo']}")
            except Exception as e:
                print("Gagal menghitung pengurangan saldo:", e)

    # === tampilkan tulisan saldo berkurang selama 3 detik ===
    if show_saldo_time and current_time - show_saldo_time < 3:
        draw_fancy_text(frame, f"Saldo Anda berkurang {total_harga_semua}", (w//2 - 200, h//2 + 100), 1.2, 2)
    elif show_saldo_time and current_time - show_saldo_time >= 3:
        pesanan = []
        last_sibi = ""
        last_kuantitas = ""
        summary_start_time = None
        pending_repeat_warning_time = None
        mode = "MAKANAN"
        data_saved = False
        show_saldo_time = None

    cv2.imshow("Sistem Pemesanan", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

engine.say("terima kasih")
engine.runAndWait()
cap.release()
cv2.destroyAllWindows()
