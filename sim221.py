import socket
import numpy as np
import threading
import struct
import time
import pandas as pd
import os

# --- AYARLAR ---
PC_IP = '192.168.1.103'
STM32_F401_IP = '192.168.1.100'
PORT_STM32_SIM_DATA = 5008
PORT_RECV_F411_DATA = 6001

STM32_OUTPUT_DIR = "./stm32_output_data"
if not os.path.exists(STM32_OUTPUT_DIR): os.makedirs(STM32_OUTPUT_DIR)

# Parametreler
TX_RATE = 2000.0
BATCH_SIZE = 50
PACKET_PERIOD = BATCH_SIZE / TX_RATE
INPUT_FFT_BINS = 160
HISTORY_LEN = 200
TOTAL_VECTOR_SIZE = INPUT_FFT_BINS + HISTORY_LEN + HISTORY_LEN

# Dosya Listesi ve İndeksler
FILE_INDICES = {
    'saglikli.csv': (40, 1000), 'eksantrik.csv': (40, 1000),
    '2ph.csv': (40, 1000), 'unb.csv': (40, 1000),
    'rotor_kirigi.csv': (0, 720), '1ph.csv': (200, 920)
}
LABELS = ['saglikli', 'rotor_kirigi', 'eksantrik', '1ph', '2ph', 'unb']


# Global Durum
class SystemState:
    def __init__(self):
        self.running = False
        self.current_label = None
        self.collected_samples = []

        # F411'den gelen verileri birleştirmek için tamponlar
        self.last_fft = np.zeros(INPUT_FFT_BINS)
        self.trq_buffer = np.zeros(HISTORY_LEN)
        self.rpm_buffer = np.zeros(HISTORY_LEN)
        self.lock = threading.Lock()


state = SystemState()


# --- F411 DINLEME THREAD'İ ---
def receiver_thread():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((PC_IP, PORT_RECV_F411_DATA))
    s.listen(1)
    print("F411 Dinleniyor...")

    while True:  # Ana döngü (Socket yeniden bağlantıları için)
        conn, addr = s.accept()
        print(f"F411 Bağlandı: {addr}")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        buf = bytearray()

        while state.running:
            try:
                chunk = conn.recv(4096)
                if not chunk: break
                buf.extend(chunk)

                while len(buf) >= 4:
                    header = struct.unpack('<I', buf[:4])[0]

                    # 1. FAST DATA (Tork/Hız) - Header: 0x1111AAAA
                    if header == 0x1111AAAA:
                        pkt_size = 404  # 4 + 200 + 200 byte
                        if len(buf) >= pkt_size:
                            pkt = buf[:pkt_size]
                            buf = buf[pkt_size:]
                            # Format: Header, 50 float Tork, 50 float Hız
                            data = struct.unpack('<I50f50f', pkt)
                            new_trq = data[1:51]
                            new_rpm = data[51:101]

                            with state.lock:
                                # Buffer'ı kaydır ve ekle
                                state.trq_buffer = np.roll(state.trq_buffer, -50)
                                state.trq_buffer[-50:] = new_trq
                                state.rpm_buffer = np.roll(state.rpm_buffer, -50)
                                state.rpm_buffer[-50:] = new_rpm
                        else:
                            break

                    # 2. SLOW DATA (FFT) - Header: 0x2222BBBB
                    elif header == 0x2222BBBB:
                        pkt_size = 4 + (INPUT_FFT_BINS * 4)  # 644 Byte
                        if len(buf) >= pkt_size:
                            pkt = buf[:pkt_size]
                            buf = buf[pkt_size:]
                            data = struct.unpack(f'<I{INPUT_FFT_BINS}f', pkt)
                            new_fft = np.array(data[1:])

                            with state.lock:
                                state.last_fft = new_fft

                                # FFT geldiğinde bir tam örnek oluşmuş demektir.
                                # [FFT (160) | TORK (200) | HIZ (200)] vektörünü kaydet
                                if state.current_label is not None:
                                    full_vector = np.concatenate([state.last_fft, state.trq_buffer, state.rpm_buffer])
                                    state.collected_samples.append(full_vector)
                        else:
                            break
                    else:
                        buf = buf[1:]  # Senkronizasyon kaybı
            except Exception as e:
                print(f"Rx Hata: {e}")
                break
        conn.close()


# --- F401 GÖNDERME ve TEST DÖNGÜSÜ ---
def run_automated_test():
    state.running = True
    # Receiver'ı başlat
    rx_t = threading.Thread(target=receiver_thread, daemon=True)
    rx_t.start()

    time.sleep(1)  # Server başlasın

    sim_header = b'\x00\x00\xAA\x42'
    cols = ['Current(PhaseA) [A]', 'Current(PhaseB) [A]', 'Current(PhaseC) [A]',
            'Moving1.Position [deg]', 'Moving1.Torque [NewtonMeter]', 'Moving1.Speed [rpm]']

    for label in LABELS:
        print(f"\n>>> TEST BAŞLIYOR: {label} <<<")
        state.current_label = label
        state.collected_samples = []  # Listeyi temizle

        # Dosyayı Oku ve Hazırla
        filename = f"{label}.csv"
        df = pd.read_csv(f"./data/{filename}")
        start, end = FILE_INDICES[filename]
        data_chunk = df[cols].iloc[start:end].values.astype(np.float32)

        # Bağlantı Kur
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.connect((STM32_F401_IP, PORT_STM32_SIM_DATA))
        except:
            print("F401'e bağlanılamadı!")
            return

        # Veri Gönder (Yaklaşık 10 saniye boyunca gönderelim ki bol örnek toplansın)
        idx = 0
        total_samples = len(data_chunk)
        start_time = time.time()

        print(f"   -> Veri F401'e basılıyor...")
        while time.time() - start_time < 15:  # 15 saniye test et
            # Kusursuz döngü (Seamless Loop)
            if idx + BATCH_SIZE <= total_samples:
                batch = data_chunk[idx: idx + BATCH_SIZE]
                idx += BATCH_SIZE
            else:
                part1 = data_chunk[idx:]
                rem = BATCH_SIZE - len(part1)
                part2 = data_chunk[:rem]
                batch = np.vstack((part1, part2))
                idx = rem

            client.sendall(sim_header + batch.tobytes())
            time.sleep(PACKET_PERIOD)

        client.close()

        # Toplanan verileri kaydet
        collected_arr = np.array(state.collected_samples)
        save_path = os.path.join(STM32_OUTPUT_DIR, f"stm32_{label}.npy")
        np.save(save_path, collected_arr)
        print(f"   -> {len(collected_arr)} örnek toplandı ve '{save_path}' konumuna kaydedildi.")

        time.sleep(2)  # Modlar arası bekleme

    state.running = False
    print("\n--- TÜM TESTLER TAMAMLANDI ---")


if __name__ == "__main__":
    run_automated_test()