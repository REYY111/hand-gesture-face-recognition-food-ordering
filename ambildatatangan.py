import csv
import mediapipe as mp
import cv2
import time

SAMPLES_PER_LETTER = 100
CSV_FILENAME = "kuantitas.csv"

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Input makanan terlebih dahulu
letter = input("Masukkan makanan yang ingin direkam (a-z): ").upper()
input("Tekan ENTER untuk mulai merekam dan membuka kamera...")  # Konfirmasi sebelum buka kamera
print(f"Silakan tunjukkan makanan '{letter}' di depan kamera...")

# Baru buka kamera
cap = cv2.VideoCapture(0)
data = []
saved_count = 0

while cap.isOpened() and saved_count < SAMPLES_PER_LETTER:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            if len(landmarks) == 21:
                flat_landmarks = []
                for lm in landmarks:
                    flat_landmarks.extend([lm[0], lm[1], lm[2]])
                data.append([letter] + flat_landmarks)
                
                saved_count += 1
                print(f"Sample {saved_count}/{SAMPLES_PER_LETTER} direkam.")
                time.sleep(0.2)
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.putText(frame, f'{letter} | Sample: {saved_count}/{SAMPLES_PER_LETTER}', (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("SIBI Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()

# Simpan data ke file CSV
if data:
    header = ['label'] + [f'{coord}{i}' for i in range(21) for coord in 'xyz']
    import os

    file_exists = os.path.isfile(CSV_FILENAME)

    with open(CSV_FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)  # Tulis header cuma kalau file belum ada
        writer.writerows(data)

    print(f"{saved_count} sample disimpan ke {CSV_FILENAME}")
else:
    print("Tidak ada data disimpan.")

