import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# === Baca Data ===
df = pd.read_csv('data_wajah.csv')
labels = df.iloc[:, 0]
features = df.iloc[:, 1:]

# === Normalisasi ===
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# ===== RANDOM FOREST =====
start_rf = time.time()
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
end_rf = time.time()

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_duration = end_rf - start_rf

# Confusion Matrix RF
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Perhitungan Per Class RF
rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_test, y_pred_rf, labels=rf_model.classes_)

# ===== Plot Bar: Precision, Recall, F1 Random Forest =====
rf_metrics_df = pd.DataFrame({
    'Precision': rf_precision,
    'Recall': rf_recall,
    'F1-Score': rf_f1
}, index=rf_model.classes_)

rf_metrics_df.plot(kind='bar', figsize=(10, 6), ylim=(0, 1), colormap='Pastel1')
plt.title("Random Forest - Precision, Recall, F1-Score per Class")
plt.ylabel("Score")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===== NEURAL NETWORK =====
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
y_train_ohe = tf.keras.utils.to_categorical(y_train_enc)
y_test_ohe = tf.keras.utils.to_categorical(y_test_enc)

nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(features.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n[Neural Network] Melatih model...")
start_nn = time.time()
history = nn_model.fit(X_train, y_train_ohe, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
loss, acc = nn_model.evaluate(X_test, y_test_ohe, verbose=0)
end_nn = time.time()

nn_accuracy = acc
nn_duration = end_nn - start_nn

# Prediksi NN
y_pred_nn_proba = nn_model.predict(X_test)
y_pred_nn = le.inverse_transform(tf.argmax(y_pred_nn_proba, axis=1).numpy())

# Perhitungan Per Class NN
nn_precision, nn_recall, nn_f1, _ = precision_recall_fscore_support(y_test, y_pred_nn, labels=le.classes_)

# ===== Plot Akurasi dan Loss NN =====
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Neural Network Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Neural Network Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ===== SIMPAN KE EXCEL DUA SHEET =====
hasil_df = pd.DataFrame({
    "Model": ["Random Forest", "Neural Network"],
    "Akurasi": [rf_accuracy, nn_accuracy],
    "Waktu Training (detik)": [rf_duration, nn_duration],
    "Waktu Training (ms)": [rf_duration * 1000, nn_duration * 1000]
})

hasil_perkelas_df = pd.DataFrame({
    "Nama": rf_model.classes_,
    "Akurasi_RF (Precision)": rf_precision,
    "Akurasi_NN (Precision)": nn_precision,
    "Waktu_RF (ms)": [rf_duration * 1000] * len(rf_model.classes_),
    "Waktu_NN (ms)": [nn_duration * 1000] * len(le.classes_)
})

with pd.ExcelWriter("hasil_evaluasi_model.xlsx") as writer:
    hasil_df.to_excel(writer, sheet_name="Ringkasan Model", index=False)
    hasil_perkelas_df.to_excel(writer, sheet_name="Akurasi per Kelas", index=False)

# ===== CETAK DI TERMINAL =====
pd.set_option("display.float_format", "{:.2f}".format)
print("\n=== Akurasi per Kelas ===")
print(hasil_perkelas_df.to_string(index=False))

print("\n>> Semua hasil disimpan ke 'hasil_evaluasi_model.xlsx' dalam dua sheet.")
