import cv2
import pickle
import numpy as np
import os




video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces_data = []
labels = []

name = input("Masukkan Nama Anda: ")
count = 0
total_capture = 100

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (50, 50))

        if count < total_capture:
            faces_data.append(face_resized)
            labels.append(name)
            count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{count}/{total_capture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture Wajah", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= total_capture:
        break

video.release()
cv2.destroyAllWindows()

# Gabungkan dengan data lama
if os.path.exists("faces_data.pickle"):
    with open("faces_data.pickle", "rb") as f:
        old_faces, old_labels = pickle.load(f)
    faces_data = old_faces + faces_data
    labels = old_labels + labels

# Simpan semua data secara utuh
with open("faces_data.pickle", "wb") as f:
    pickle.dump((faces_data, labels), f)

print(f"✅ Data wajah untuk '{name}' berhasil disimpan.")
