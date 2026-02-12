import socket
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets, QtGui
import threading
import struct
import time
import pandas as pd
import glob
import os

# --- AĞ AYARLARI ---
PC_IP = '192.168.1.103'
STM32_F401_IP = '192.168.1.100'

PORT_STM32_SIM_DATA = 5008
PORT_RECV_F401_FFT = 5007
PORT_RECV_F411_DATA = 6001

# --- FREKANS AYARLARI ---
TX_RATE = 250.0
BATCH_SIZE = 50
SAMPLE_PERIOD = 1.0 / TX_RATE
PACKET_PERIOD = BATCH_SIZE * SAMPLE_PERIOD

SIGNAL_FS = 2000.0
FFT_FULL_SIZE = 2048
FFT_OUTPUT_SIZE = 1024
FREQ_RES = SIGNAL_FS / FFT_FULL_SIZE  # Yaklaşık 1.95 Hz

# --- YENİ AYARLAR ---
HISTORY_SIZE = 400  # Tork ve Hız için geçmiş boyutu
PARTIAL_FFT_SIZE = 160  # F411'den gelen FFT boyutu

# --- SPEKTROGRAM AYARLARI ---
SPEC_HISTORY_LEN = 400  # Zaman ekseninde ne kadar veri tutulacak

# --- PAKET YAPILARI ---
FAST_FMT = '<I50f50f'
FAST_SIZE = struct.calcsize(FAST_FMT)

SLOW_FMT = f'<I{PARTIAL_FFT_SIZE}f'
SLOW_SIZE = struct.calcsize(SLOW_FMT)

FFT_PACKET_SIZE = FFT_OUTPUT_SIZE * 4


class SimulationWorker(QtCore.QThread):
    status_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.current_df = None
        self.data_folder = "./data"
        self.columns = ['Current(PhaseA) [A]', 'Current(PhaseB) [A]', 'Current(PhaseC) [A]', 'Moving1.Position [deg]',
                        'Moving1.Torque [NewtonMeter]', 'Moving1.Speed [rpm]']
        self.sim_header = b'\x00\x00\xAA\x42'

    def set_state(self, filename):
        filepath = os.path.join(self.data_folder, filename)
        if os.path.exists(filepath):
            self.current_df = pd.read_csv(filepath)
            self.status_signal.emit(f"Yüklendi: {filename}")

    def run(self):
        self.running = True
        while self.running:
            try:
                self.status_signal.emit(f"Bağlanıyor -> {STM32_F401_IP}...")
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.connect((STM32_F401_IP, PORT_STM32_SIM_DATA))
                self.status_signal.emit("Bağlantı Başarılı!")

                if self.current_df is not None:
                    df = self.current_df
                    num_rows = len(df)
                    idx = 0
                    next_call = time.perf_counter()

                    while self.running and idx < num_rows:
                        if self.current_df is not df: break

                        end_idx = min(idx + BATCH_SIZE, num_rows)
                        batch_rows = df.iloc[idx:end_idx]

                        payload_bytes = b''.join(
                            [struct.pack('<6f', *row[self.columns]) for _, row in batch_rows.iterrows()])

                        if len(batch_rows) == BATCH_SIZE:
                            client.sendall(self.sim_header + payload_bytes)

                        idx += BATCH_SIZE
                        if idx >= num_rows: idx = 0

                        next_call += PACKET_PERIOD
                        while time.perf_counter() < next_call:
                            pass
                else:
                    time.sleep(1)
            except Exception as e:
                self.status_signal.emit(f"Hata: {e}")
                time.sleep(2)
            finally:
                client.close()

    def stop(self):
        self.running = False;
        self.wait()


class MotorMonitor:
    def __init__(self):
        self.running = False
        self.lock = threading.Lock()

        # --- VERİ TAMPONLARI ---
        self.full_iq_fft = None

        # Geçmiş verisi için bufferları sıfır ile başlat
        self.trq_history = np.zeros(HISTORY_SIZE)
        self.rpm_history = np.zeros(HISTORY_SIZE)

        self.f411_fft = None

        # Frekans eksenleri
        self.freqs_full = np.arange(FFT_OUTPUT_SIZE) * FREQ_RES
        self.freqs_partial = np.arange(PARTIAL_FFT_SIZE) * FREQ_RES

        # --- SPEKTROGRAM BUFFER ---
        # (Zaman, Frekans) formatında buffer oluşturuyoruz.
        # Varsayılan değer olarak gürültü tabanı (-140 dB) veriyoruz.
        self.spec_buf = np.full((SPEC_HISTORY_LEN, FFT_OUTPUT_SIZE), -140, dtype=np.float32)

        # Renk haritası (rms_plotter'dan alındı)
        pos = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        color = np.array([
            [0, 0, 0, 255],  # Siyah
            [30, 0, 70, 255],  # Koyu Mor
            [180, 40, 60, 255],  # Kırmızımsı
            [250, 150, 0, 255],  # Turuncu
            [255, 255, 200, 255]  # Açık Sarı/Beyaz
        ], dtype=np.ubyte)
        self.colormap = pg.ColorMap(pos, color)

        self.sim_worker = SimulationWorker()
        self.app = QtWidgets.QApplication([])
        self.setup_gui()
        self.sim_worker.status_signal.connect(self.update_sim_log)

    def setup_gui(self):
        self.main_window = QtWidgets.QWidget()
        self.main_window.setWindowTitle('STM32 Monitor')
        self.main_window.resize(1200, 800)
        self.main_window.setStyleSheet("background-color: #121212; color: #e0e0e0;")
        layout = QtWidgets.QVBoxLayout(self.main_window)

        self.btn_start = QtWidgets.QPushButton("SİSTEMİ BAŞLAT")
        self.btn_start.setStyleSheet("background-color: #2E7D32; font-weight: bold; padding: 10px;")
        self.btn_start.clicked.connect(self.start_system)
        layout.addWidget(self.btn_start)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.setStyleSheet("""
                    QTabWidget::pane { border: 1px solid #333; }
                    QTabBar::tab { background: #222; color: #888; padding: 8px 16px; }
                    QTabBar::tab:selected { background: #444; color: #2196F3; font-weight: bold; }
                """)

        # --- TAB 1: FULL FFT (F401) ---
        self.tab_fft = pg.GraphicsLayoutWidget()
        self.tabs.addTab(self.tab_fft, "Iq Full FFT")
        p1 = self.tab_fft.addPlot(title="Full Spectrum (0-1000 Hz)")
        self.curve_full_fft = p1.plot(pen='#00E676', fillLevel=-140, brush=(0, 230, 118, 50))
        p1.setLabel('bottom', 'Frekans (Hz)')
        p1.setLabel('left', 'Genlik (dB)')
        p1.showGrid(x=True, y=True)
        p1.setXRange(0, 1000)
        p1.setYRange(-20, 100)

        # --- TAB 2: GATEWAY (F411) ---
        self.tab_f411 = pg.GraphicsLayoutWidget()
        self.tabs.addTab(self.tab_f411, "F411 Verileri")

        # Tork Grafiği
        p_trq = self.tab_f411.addPlot(title=f"Tork (Son {HISTORY_SIZE} Örnek)", row=0, col=0)
        self.curve_trq = p_trq.plot(pen='#FFC107')
        p_trq.showGrid(x=True, y=True)

        # RPM Grafiği
        p_rpm = self.tab_f411.addPlot(title=f"RPM (Son {HISTORY_SIZE} Örnek)", row=0, col=1)
        self.curve_rpm = p_rpm.plot(pen='#E040FB')
        p_rpm.showGrid(x=True, y=True)

        self.tab_f411.nextRow()

        # Kısmi FFT Grafiği
        p_part = self.tab_f411.addPlot(title="F411 FFT ", row=1, col=0, colspan=2)
        self.curve_part_fft = p_part.plot(pen='#29B6F6', fillLevel=-140, brush=(41, 182, 246, 50))
        p_part.setLabel('bottom', 'Frekans (Hz)')
        p_part.showGrid(x=True, y=True)
        p_part.setXRange(0, 160)
        p_part.setYRange(-20, 100)

        # --- TAB 3: SIMULASYON ---
        self.tab_sim = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_sim, "Simülasyon")
        self._setup_sim_tab()

        # --- TAB 4: SPEKTROGRAM (YENİ EKLENDİ) ---
        self.tab_spec = pg.GraphicsLayoutWidget()
        self.tabs.addTab(self.tab_spec, "Spektrogram")

        p_spec = self.tab_spec.addPlot(title="Full FFT Spektrogramı")
        p_spec.setLabel('bottom', "Zaman", units='frame')
        p_spec.setLabel('left', "Frekans", units='Hz')
        p_spec.setYRange(0, 1000)  # 0-1000 Hz arası görünüm

        self.img_item_spec = pg.ImageItem()
        p_spec.addItem(self.img_item_spec)

        # Y eksenini Hz cinsine ölçeklemek için transform
        tr = QtGui.QTransform()
        tr.scale(1, FREQ_RES)  # X ekseni (zaman) 1 birim, Y ekseni (frekans) çözünürlük kadar
        self.img_item_spec.setTransform(tr)

        # Renk skalası (HistogramLUT)
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img_item_spec)
        hist.gradient.setColorMap(self.colormap)
        hist.setLevels(-120, 20)  # Renk aralığı ayarı (dB)
        self.tab_spec.addItem(hist)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)  # Güncelleme hızı
        self.main_window.show()

    def _setup_sim_tab(self):
        l = QtWidgets.QVBoxLayout(self.tab_sim)
        h = QtWidgets.QHBoxLayout()
        self.combo_files = QtWidgets.QComboBox()
        btn = QtWidgets.QPushButton("Yenile")
        btn.clicked.connect(self.refresh_files)
        h.addWidget(self.combo_files)
        h.addWidget(btn)
        l.addLayout(h)
        h2 = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Başlat")
        b1.clicked.connect(self.start_sim_worker)
        b2 = QtWidgets.QPushButton("Durdur")
        b2.clicked.connect(self.stop_sim_worker)
        h2.addWidget(b1)
        h2.addWidget(b2)
        l.addLayout(h2)
        self.sim_log = QtWidgets.QTextEdit()
        l.addWidget(self.sim_log)
        self.refresh_files()

    def refresh_files(self):
        self.combo_files.clear()
        if not os.path.exists('./data'): os.makedirs('./data')
        self.combo_files.addItems([os.path.basename(f) for f in glob.glob('./data/*.csv')])

    def start_sim_worker(self):
        if self.combo_files.currentText():
            self.sim_worker.set_state(self.combo_files.currentText())
            self.sim_worker.start()

    def stop_sim_worker(self):
        self.sim_worker.stop()

    def update_sim_log(self, msg):
        self.sim_log.append(msg)

    def server_thread_f401_fft(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((PC_IP, PORT_RECV_F401_FFT))
            s.listen(1)
            while self.running:
                conn, addr = s.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                buf = bytearray()
                while self.running:
                    chunk = conn.recv(8192)
                    if not chunk: break
                    buf.extend(chunk)
                    while len(buf) >= FFT_PACKET_SIZE:
                        pkt = buf[:FFT_PACKET_SIZE]
                        buf = buf[FFT_PACKET_SIZE:]
                        arr = np.frombuffer(pkt, dtype='<f4')
                        with self.lock: self.full_iq_fft = arr
                conn.close()
        except Exception as e:
            print(f"F401 Err: {e}")

    def server_thread_f411_data(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((PC_IP, PORT_RECV_F411_DATA))
            s.listen(1)
            print(f"F411 Dinleniyor: {PORT_RECV_F411_DATA}")

            while self.running:
                conn, addr = s.accept()
                print(f"F411 BAĞLANDI! {addr}")
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                buf = bytearray()
                while self.running:
                    try:
                        chunk = conn.recv(4096)
                        if not chunk: break
                        buf.extend(chunk)

                        while len(buf) >= 4:
                            header = struct.unpack('<I', buf[:4])[0]

                            if header == 0x1111AAAA:  # --- FAST DATA (Tork/Hız) ---
                                if len(buf) >= FAST_SIZE:
                                    pkt = buf[:FAST_SIZE]
                                    buf = buf[FAST_SIZE:]
                                    data = struct.unpack(FAST_FMT, pkt)

                                    new_trq = np.array(data[1:51])
                                    new_rpm = np.array(data[51:101])

                                    with self.lock:
                                        self.trq_history = np.roll(self.trq_history, -50)
                                        self.trq_history[-50:] = new_trq

                                        self.rpm_history = np.roll(self.rpm_history, -50)
                                        self.rpm_history[-50:] = new_rpm
                                else:
                                    break

                            elif header == 0x2222BBBB:  # --- SLOW DATA (FFT) ---
                                if len(buf) >= SLOW_SIZE:
                                    pkt = buf[:SLOW_SIZE]
                                    buf = buf[SLOW_SIZE:]
                                    data = struct.unpack(SLOW_FMT, pkt)
                                    with self.lock:
                                        self.f411_fft = np.array(data[1:])
                                else:
                                    break
                            else:
                                buf = buf[1:]  # Senkronizasyon kaybı, 1 bayt ilerle
                    except Exception as e:
                        print(f"Paket İşleme Hatası: {e}")
                        break
                conn.close()
        except Exception as e:
            print(f"F411 Server Hatası: {e}")

    def start_system(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.server_thread_f401_fft, daemon=True).start()
            threading.Thread(target=self.server_thread_f411_data, daemon=True).start()
            self.btn_start.setEnabled(False)
            self.btn_start.setText("SUNUCULAR AKTİF")

    def update_plots(self):
        d_full, d_trq_hist, d_rpm_hist, d_part = None, None, None, None

        with self.lock:
            # Full FFT Kopyala
            if self.full_iq_fft is not None:
                d_full = self.full_iq_fft.copy()

            # Geçmiş Verilerini Kopyala
            d_trq_hist = self.trq_history.copy()
            d_rpm_hist = self.rpm_history.copy()

            # Partial FFT Kopyala
            if self.f411_fft is not None:
                d_part = self.f411_fft.copy()

        # 1. Full FFT Çizimi
        if d_full is not None:
            try:
                # dB hesaplama
                d_full_db = 20 * np.log10(d_full + 1e-9)
                self.curve_full_fft.setData(self.freqs_full, d_full_db)

                # --- SPEKTROGRAM GÜNCELLEME (YENİ) ---
                # 1. Buffer'ı sola kaydır (En eski veri [0] silinir, hepsi 1 sola kayar)
                self.spec_buf = np.roll(self.spec_buf, -1, axis=0)

                # 2. Yeni veriyi en son satıra ekle (En sağa)
                self.spec_buf[-1] = d_full_db

                # 3. Görüntüyü güncelle
                self.img_item_spec.setImage(self.spec_buf, autoLevels=False)
            except Exception as e:
                # print(f"Spec Err: {e}")
                pass

        # 2. Tork ve RPM Çizimi (Geçmiş Verisiyle)
        try:
            x_axis = np.arange(HISTORY_SIZE)
            self.curve_trq.setData(x_axis, d_trq_hist)
            self.curve_rpm.setData(x_axis, d_rpm_hist)
        except Exception as e:
            print(f"Plot Err: {e}")

        # 3. Kısmi FFT Çizimi
        if d_part is not None:
            try:
                if len(d_part) == len(self.freqs_partial):
                    d_part_db = 20 * np.log10(d_part + 1e-9)
                    self.curve_part_fft.setData(self.freqs_partial, d_part_db)
            except:
                pass


if __name__ == "__main__":
    app = MotorMonitor()
    QtWidgets.QApplication.instance().exec()