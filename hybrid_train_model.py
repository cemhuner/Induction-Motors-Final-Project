import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import get_window
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input, Model
from itertools import cycle

# --- AYARLAR ---
DATA_FOLDER = "./data"
REPORT_DIR = "./hybrid_training_report"
if not os.path.exists(REPORT_DIR): os.makedirs(REPORT_DIR)

FFT_SIZE = 2048
INPUT_FFT_BINS = 160
HISTORY_LEN = 200
TOTAL_INPUT_SIZE = INPUT_FFT_BINS + HISTORY_LEN + HISTORY_LEN  # 560

FILE_INDICES = {
    'saglikli.csv': (40, 1000), 'eksantrik.csv': (40, 1000),
    '2ph.csv': (40, 1000), 'unb.csv': (40, 1000),
    'rotor_kirigi.csv': (0, 720), '1ph.csv': (200, 920)
}
LABELS = ['saglikli', 'rotor_kirigi', 'eksantrik', '1ph', '2ph', 'unb']


# --- 1. VERİ HAZIRLAMA (Gürültülü & Güvenli) ---
def load_and_prep_data(n_samples_per_class=2000):
    X = []
    y = []
    stats = {'t_min': float('inf'), 't_max': float('-inf'),
             'r_min': float('inf'), 'r_max': float('-inf'), 'fft_max': float('-inf')}

    print("--- Veri Yükleniyor ve Hazırlanıyor ---")

    for label in LABELS:
        filename = f"{label}.csv"
        filepath = os.path.join(DATA_FOLDER, filename)
        if not os.path.exists(filepath): continue

        df = pd.read_csv(filepath)
        if df.isnull().values.any(): df = df.fillna(method='ffill').fillna(0)

        start, end = FILE_INDICES.get(filename, (0, len(df)))
        curr_raw = df['Current(PhaseA) [A]'].values[start:end]
        trq_raw = df['Moving1.Torque [NewtonMeter]'].values[start:end]
        spd_raw = df['Moving1.Speed [rpm]'].values[start:end]

        # Looping
        stride = 2
        required_len = FFT_SIZE + (n_samples_per_class * stride)
        while len(curr_raw) < required_len:
            curr_raw = np.concatenate((curr_raw, curr_raw))
            trq_raw = np.concatenate((trq_raw, trq_raw))
            spd_raw = np.concatenate((spd_raw, spd_raw))

        try:
            window = get_window('hann', FFT_SIZE)
        except:
            window = np.hanning(FFT_SIZE)

        for i in range(n_samples_per_class):
            idx = i * stride

            # Gürültü Ekleme
            noise_level = 0.005  # %0.5 gürültü
            segment_curr = curr_raw[idx: idx + FFT_SIZE]
            segment_curr += np.random.normal(0, noise_level * np.std(segment_curr), len(segment_curr))

            # FFT
            fft_val = np.abs(np.fft.rfft(segment_curr * window)) / FFT_SIZE
            feat_fft = fft_val[:INPUT_FFT_BINS]

            # Tork/Hız
            end_ptr = idx + FFT_SIZE
            feat_trq = trq_raw[end_ptr - HISTORY_LEN: end_ptr]
            feat_rpm = spd_raw[end_ptr - HISTORY_LEN: end_ptr]

            # İstatistik
            stats['t_min'] = min(stats['t_min'], feat_trq.min())
            stats['t_max'] = max(stats['t_max'], feat_trq.max())
            stats['r_min'] = min(stats['r_min'], feat_rpm.min())
            stats['r_max'] = max(stats['r_max'], feat_rpm.max())
            stats['fft_max'] = max(stats['fft_max'], feat_fft.max())

            combined = np.concatenate([feat_fft, feat_trq, feat_rpm])
            X.append(combined)
            y.append(LABELS.index(label))

    X = np.array(X)
    y = np.array(y)

    # Normalizasyon
    if stats['fft_max'] == 0: stats['fft_max'] = 1.0
    X[:, :INPUT_FFT_BINS] /= stats['fft_max']

    t_range = stats['t_max'] - stats['t_min'] if stats['t_max'] != stats['t_min'] else 1.0
    t_start = INPUT_FFT_BINS
    t_end = t_start + HISTORY_LEN
    X[:, t_start:t_end] = (X[:, t_start:t_end] - stats['t_min']) / t_range

    r_range = stats['r_max'] - stats['r_min'] if stats['r_max'] != stats['r_min'] else 1.0
    r_start = t_end
    r_end = r_start + HISTORY_LEN
    X[:, r_start:r_end] = (X[:, r_start:r_end] - stats['r_min']) / r_range

    X = np.nan_to_num(X, nan=0.0)
    return X, y, stats


# --- YARDIMCI: VERİYİ PARÇALAMA ---
def split_inputs(X):
    """
    Tek parça olan X vektörünü (560,) CNN ve MLP için ikiye böler.
    1. FFT: (160,) -> CNN için (160, 1) yapılır.
    2. Tork/Hız: (400,) -> MLP için düz bırakılır.
    """
    X_fft = X[:, :INPUT_FFT_BINS]
    # CNN 3 boyutlu veri ister: (Batch, Steps, Channels) -> (N, 160, 1)
    X_fft = X_fft.reshape(-1, INPUT_FFT_BINS, 1)

    X_ts = X[:, INPUT_FFT_BINS:]  # Geriye kalan 400 veri (Tork + Hız)

    return [X_fft, X_ts]


# --- 2. HİBRİT MODEL MİMARİSİ (CNN + MLP) ---
def create_hybrid_model(num_classes):
    # --- KOL 1: CNN (FFT Verisi İçin) ---
    input_fft = Input(shape=(INPUT_FFT_BINS, 1), name='fft_input')

    # Conv1D: Frekans desenlerini yakalar
    # Filter 16, Kernel 3: Çok hafif tutuyoruz (STM32 dostu)
    x1 = layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(input_fft)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)  # Boyutu yarıya indir (80'e düşer)

    x1 = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)  # Boyutu yarıya indir (40'a düşer)

    x1 = layers.Flatten()(x1)  # Düzleştir
    x1 = layers.Dropout(0.2)(x1)  # Ezberlemeyi önle

    # --- KOL 2: MLP (Tork ve Hız İçin) ---
    input_ts = Input(shape=(HISTORY_LEN + HISTORY_LEN,), name='ts_input')

    x2 = layers.Dense(32, activation='relu')(input_ts)

    # --- BİRLEŞTİRME (FUSION) ---
    combined = layers.concatenate([x1, x2])

    # Son Karar Katmanı
    z = layers.Dense(32, activation='relu')(combined)
    outputs = layers.Dense(num_classes, activation='softmax')(z)

    model = Model(inputs=[input_fft, input_ts], outputs=outputs, name="Hybrid_F411")
    return model


# --- 3. RAPORLAMA (Hibrit Uyumlu) ---
def generate_reports(model, history, X_test_split, y_test, class_names):
    print("\n--- Raporlar Oluşturuluyor ---")

    # Tahmin
    y_pred_probs = model.predict(X_test_split)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 1. Loss/Accuracy Grafiği
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss Eğrisi')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy Eğrisi')
    plt.legend()
    plt.savefig(f"{REPORT_DIR}/training_curves.png")
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.savefig(f"{REPORT_DIR}/confusion_matrix.png")
    plt.close()

    # 3. Metin Raporu
    report = classification_report(y_test, y_pred, target_names=class_names)
    with open(f"{REPORT_DIR}/results.txt", "w") as f:
        f.write(report)
        f.write(f"\nTest Accuracy: {np.mean(y_test == y_pred):.4f}")

    print(f"Raporlar '{REPORT_DIR}' klasörüne kaydedildi.")


# --- ANA AKIŞ ---
# 1. Veri Hazırla
X, y, stats = load_and_prep_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Veriyi Modele Uygun Hale Getir (Parçala)
# Keras Functional API için girdileri liste olarak vermeliyiz: [FFT_Girdisi, TS_Girdisi]
X_train_split = split_inputs(X_train)
X_test_split = split_inputs(X_test)

# 3. Model Kurulumu
model = create_hybrid_model(len(LABELS))
model.summary()  # Parametre sayısını konsolda görebilirsin

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Eğitim
print("\n--- Hibrit Model Eğitiliyor ---")
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_split, y_train,  # Girişler liste formatında
    epochs=30,
    batch_size=32,
    validation_data=(X_test_split, y_test),
    callbacks=[early_stop],
    verbose=1
)

# 5. Kayıt ve Raporlama
model.save("stm32_hybrid_cnn_mlp.h5")
generate_reports(model, history, X_test_split, y_test, LABELS)

# 6. C Kodu Çıktısı
print("\n" + "=" * 50)
print("STM32 C KODU İÇİN GEREKLİ MAKROLAR")
print("=" * 50)
print(f"#define NORM_FFT_MAX    {stats['fft_max']:.6f}f")
print(f"#define NORM_TRQ_MIN    {stats['t_min']:.6f}f")
print(f"#define NORM_TRQ_MAX    {stats['t_max']:.6f}f")
print(f"#define NORM_RPM_MIN    {stats['r_min']:.6f}f")
print(f"#define NORM_RPM_MAX    {stats['r_max']:.6f}f")
print("=" * 50)