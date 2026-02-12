import numpy as np
import matplotlib.pyplot as plt
import os
import json

# --- AYARLAR ---
REF_DIR = "./reference_data"
STM32_DIR = "./stm32_output_data"
LABELS = ['saglikli', 'rotor_kirigi', 'eksantrik', '1ph', '2ph', 'unb']

# Vektör Boyutları
FFT_LEN = 160
TRQ_LEN = 200
RPM_LEN = 200
FFT_SIZE = 2048  # F401'deki FFT Pencere boyutu

# İstatistikleri Yükle
stats_path = os.path.join(REF_DIR, "stats.json")
if not os.path.exists(stats_path):
    print("HATA: 'reference_data/stats.json' bulunamadı! Önce eğitim kodunu çalıştırın.")
    exit()

with open(stats_path, "r") as f:
    stats = json.load(f)


def normalize_vector(vec, stats, is_stm32=False):
    # Vektörü Parçala
    fft = vec[:FFT_LEN]
    trq = vec[FFT_LEN: FFT_LEN + TRQ_LEN]
    rpm = vec[FFT_LEN + TRQ_LEN:]

    # --- KRİTİK DÜZELTME: STM32 ÖLÇEKLEME ---
    if is_stm32:
        # STM32 FFT sonucu ham toplamdır, Python ise ortalamadır.
        # Bu yüzden STM32 verisini FFT boyutuna bölerek Python seviyesine indiriyoruz.
        fft = fft / FFT_SIZE

    # 1. FFT Normalizasyonu
    if stats['fft_max'] == 0: stats['fft_max'] = 1.0
    fft_norm = fft / stats['fft_max']

    # 2. Tork Normalizasyonu
    t_range = stats['t_max'] - stats['t_min']
    if t_range == 0: t_range = 1.0
    trq_norm = (trq - stats['t_min']) / t_range

    # 3. Hız Normalizasyonu
    r_range = stats['r_max'] - stats['r_min']
    if r_range == 0: r_range = 1.0
    rpm_norm = (rpm - stats['r_min']) / r_range

    return np.concatenate([fft_norm, trq_norm, rpm_norm])


def compare_signals_fixed():
    print("=== DÜZELTİLMİŞ DOĞRULAMA VE KIYASLAMA ===")

    for label in LABELS:
        ref_path = os.path.join(REF_DIR, f"ref_{label}.npy")
        stm_path = os.path.join(STM32_DIR, f"stm32_{label}.npy")

        if not os.path.exists(ref_path) or not os.path.exists(stm_path):
            continue

        # Ham verileri yükle
        ref_raw_all = np.load(ref_path)
        stm_raw_all = np.load(stm_path)

        if len(stm_raw_all) == 0:
            print(f"{label}: STM32 verisi yok!")
            continue

        # Gürültüyü azaltmak için tüm örneklerin ortalamasını al
        ref_vec_avg = np.mean(ref_raw_all, axis=0)
        stm_vec_avg = np.mean(stm_raw_all, axis=0)

        # --- NORMALİZASYON (DÜZELTME İLE) ---
        # Referans veri zaten Python ölçeğinde olduğu için is_stm32=False
        ref_norm = normalize_vector(ref_vec_avg, stats, is_stm32=False)

        # STM32 verisi büyük ölçekte olduğu için is_stm32=True (Bölme işlemi yapılacak)
        stm_norm = normalize_vector(stm_vec_avg, stats, is_stm32=True)

        # --- BENZERLİK ANALİZİ ---
        # Cosine Similarity (Vektörlerin yönü ne kadar benziyor?)
        dot_product = np.dot(ref_norm, stm_norm)
        norm_a = np.linalg.norm(ref_norm)
        norm_b = np.linalg.norm(stm_norm)
        similarity = dot_product / (norm_a * norm_b)

        # Euclidean Distance (Noktalar birbirine ne kadar yakın?)
        mse = np.mean((ref_norm - stm_norm) ** 2)

        print(f"\n>>> {label.upper()} ANALİZİ <<<")
        print(f"   Benzerlik (Cosine): {similarity:.4f} (Hedef > 0.95)")
        print(f"   Hata (MSE): {mse:.6f} (Hedef < 0.01)")

        # --- GRAFİK ÇİZİMİ ---
        plt.figure(figsize=(12, 8))

        # FFT Kısmı
        plt.subplot(3, 1, 1)
        plt.plot(ref_norm[:FFT_LEN], label='Python Referans', color='blue', linewidth=2, alpha=0.7)
        plt.plot(stm_norm[:FFT_LEN], label='STM32 F411', color='red', linestyle='--', linewidth=2, alpha=0.9)
        plt.title(f"{label} - FFT Spektrumu (Normalize)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Tork Kısmı
        plt.subplot(3, 1, 2)
        plt.plot(ref_norm[FFT_LEN:FFT_LEN + TRQ_LEN], label='Python', color='green')
        plt.plot(stm_norm[FFT_LEN:FFT_LEN + TRQ_LEN], label='STM32', color='orange', linestyle='--')
        plt.title("Tork Sinyali")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Hız Kısmı
        plt.subplot(3, 1, 3)
        plt.plot(ref_norm[-RPM_LEN:], label='Python', color='purple')
        plt.plot(stm_norm[-RPM_LEN:], label='STM32', color='magenta', linestyle='--')
        plt.title("Hız Sinyali")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    compare_signals_fixed()