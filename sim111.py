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
import datetime

# --- DEBUG AYARI ---
DEBUG_MODE = True  # Konsol çıktılarını görmek için True yapın


def log_debug(msg):
    if DEBUG_MODE:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {msg}")


# --- AĞ AYARLARI ---
# '0.0.0.0' tüm ağ arayüzlerini dinler (Ethernet, WiFi vs.)
# Bu sayede IP değişse bile kod çalışmaya devam eder.
PC_IP = '0.0.0.0'

# F401'in IP'si (Simülasyon verisini buraya yolluyoruz)
STM32_F401_IP = '192.168.1.100'

PORT_STM32_SIM_DATA = 5008  # PC -> F401 (Simülasyon Verisi)
PORT_RECV_F401_FFT = 5007  # F401 -> PC (Ham FFT - Opsiyonel)
PORT_RECV_F411_DATA = 6001  # F411 -> PC (AI Sonuçları + Veri)

# --- SİMÜLASYON AYARLARI ---
TX_RATE = 250.0
BATCH_SIZE = 50
SAMPLE_PERIOD = 1.0 / TX_RATE
PACKET_PERIOD = BATCH_SIZE * SAMPLE_PERIOD

SIGNAL_FS = 2000.0
FFT_FULL_SIZE = 2048
FFT_OUTPUT_SIZE = 1024
FREQ_RES = SIGNAL_FS / FFT_FULL_SIZE

HISTORY_SIZE = 400
PARTIAL_FFT_SIZE = 160
SPEC_HISTORY_LEN = 400

# --- PAKET YAPILARI ---
# 1. FAST DATA (Tork ve Hız)
# Header (4) + 50 float Tork (200) + 50 float Hız (200) = 404 Byte
FAST_FMT = '<I50f50f'
FAST_SIZE = struct.calcsize(FAST_FMT)

# 2. SLOW DATA (FFT + AI Sonucu)
# Header (4) + 160 float FFT (640) + 1 uint32 AI Result (4) = 648 Byte
# Sondaki 'I' harfi AI sonucunu temsil eder.
SLOW_FMT = f'<I{PARTIAL_FFT_SIZE}fI'
SLOW_SIZE = struct.calcsize(SLOW_FMT)

FFT_PACKET_SIZE = FFT_OUTPUT_SIZE * 4

# --- ARIZA KODLARI VE RENKLER ---
FAULT_CODES = {
    0: "SAĞLIKLI (NORMAL)",
    1: "EKSANTRİK ARIZASI",
    2: "ROTOR KIRIĞI",
    3: "1 FAZ KISA DEVRE",
    4: "2 FAZ KISA DEVRE",
    5: "DENGESİZ AKIM (UNBALANCE)"
}

FAULT_COLORS = {
    0: "#2E7D32",  # Yeşil
    1: "#C62828",  # Kırmızı
    2: "#B71C1C",  # Koyu Kırmızı
    3: "#E65100",  # Turuncu
    4: "#BF360C",  # Koyu Turuncu
    5: "#F9A825"  # Sarı
}


class SimulationWorker(QtCore.QThread):
    """
    CSV dosyasındaki verileri okuyup F401'e gönderen Thread.
    """
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
            self.status_signal.emit(f"Dosya Seçildi: {filename}")
            log_debug(f"Simülasyon Dosyası Hazır: {filename}")

    def run(self):
        self.running = True
        while self.running:
            client = None
            try:
                self.status_signal.emit(f"Bağlanıyor -> {STM32_F401_IP}...")
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.settimeout(5)  # 5 saniye zaman aşımı
                client.connect((STM32_F401_IP, PORT_STM32_SIM_DATA))

                self.status_signal.emit("Bağlantı Başarılı! Veri Gönderiliyor...")
                log_debug("Simülasyon: F401'e Bağlandı.")

                if self.current_df is not None:
                    df = self.current_df
                    num_rows = len(df)
                    idx = 0
                    next_call = time.perf_counter()
                    pkt_count = 0

                    while self.running and idx < num_rows:
                        if self.current_df is not df: break  # Dosya değiştiyse çık

                        end_idx = min(idx + BATCH_SIZE, num_rows)
                        batch_rows = df.iloc[idx:end_idx]
                        payload_bytes = b''.join(
                            [struct.pack('<6f', *row[self.columns]) for _, row in batch_rows.iterrows()])

                        if len(batch_rows) == BATCH_SIZE:
                            client.sendall(self.sim_header + payload_bytes)
                            pkt_count += 1
                            if pkt_count % 200 == 0:
                                log_debug(f"Simülasyon: {pkt_count} paket gönderildi.")

                        idx += BATCH_SIZE
                        if idx >= num_rows: idx = 0  # Başa sar

                        # Zamanlama kontrolü (Gerçek zamanlı simülasyon için)
                        next_call += PACKET_PERIOD
                        sleep_time = next_call - time.perf_counter()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                else:
                    time.sleep(1)
            except Exception as e:
                self.status_signal.emit(f"Hata: {e}")
                log_debug(f"Simülasyon Hatası: {e}")
                time.sleep(2)
            finally:
                if client: client.close()

    def stop(self):
        self.running = False;
        self.wait()


class MotorMonitor:
    def __init__(self):
        self.running = False
        self.lock = threading.Lock()

        # Veri Tamponları
        self.full_iq_fft = None
        self.trq_history = np.zeros(HISTORY_SIZE)
        self.rpm_history = np.zeros(HISTORY_SIZE)
        self.f411_fft = None

        # --- AI SONUÇ DEĞİŞKENİ ---
        self.latest_fault_code = 0  # 0: Sağlıklı varsayılan

        # Grafikler İçin Eksenler
        self.freqs_full = np.arange(FFT_OUTPUT_SIZE) * FREQ_RES
        self.freqs_partial = np.arange(PARTIAL_FFT_SIZE) * FREQ_RES

        # Spektrogram Buffer
        self.spec_buf = np.full((SPEC_HISTORY_LEN, FFT_OUTPUT_SIZE), -140, dtype=np.float32)

        # Renk Haritası
        pos = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        color = np.array(
            [[0, 0, 0, 255], [30, 0, 70, 255], [180, 40, 60, 255], [250, 150, 0, 255], [255, 255, 200, 255]],
            dtype=np.ubyte)
        self.colormap = pg.ColorMap(pos, color)

        # GUI Başlat
        self.sim_worker = SimulationWorker()
        self.app = QtWidgets.QApplication([])
        self.setup_gui()
        self.sim_worker.status_signal.connect(self.update_sim_log)

    def setup_gui(self):
        self.main_window = QtWidgets.QWidget()
        self.main_window.setWindowTitle('STM32 AI Motor Teşhis Sistemi')
        self.main_window.resize(1200, 850)
        self.main_window.setStyleSheet("background-color: #1e1e1e; color: #e0e0e0;")
        layout = QtWidgets.QVBoxLayout(self.main_window)

        # Üst Panel: Başlat Butonu ve Durum
        top_layout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("SİSTEMİ BAŞLAT")
        self.btn_start.setStyleSheet(
            "background-color: #2E7D32; font-weight: bold; padding: 12px; font-size: 14px; border-radius: 5px;")
        self.btn_start.clicked.connect(self.start_system)
        top_layout.addWidget(self.btn_start)
        layout.addLayout(top_layout)

        # Sekmeler
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; }
            QTabBar::tab { background: #2c2c2c; color: #aaa; padding: 10px 20px; font-size: 12px; border-top-left-radius: 4px; border-top-right-radius: 4px;}
            QTabBar::tab:selected { background: #3d3d3d; color: #4fc3f7; font-weight: bold; border-bottom: 2px solid #4fc3f7; }
        """)

        # TAB 1: ARIZA TEŞHİS (AI SONUCU) - En Önemli Ekran
        self.tab_diag = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_diag, "AI Arıza Teşhis")
        diag_layout = QtWidgets.QVBoxLayout(self.tab_diag)

        title_lbl = QtWidgets.QLabel("MOTOR SAĞLIK DURUMU (YAPAY ZEKA)")
        title_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet("font-size: 24px; color: #BBB; margin-top: 30px; font-weight: bold;")
        diag_layout.addWidget(title_lbl)

        # Arıza Gösterge Kutusu
        self.lbl_fault_status = QtWidgets.QLabel("SİSTEM BEKLENİYOR...")
        self.lbl_fault_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_fault_status.setStyleSheet("""
            font-size: 56px; 
            font-weight: bold; 
            color: #555; 
            background-color: #1a1a1a; 
            border: 2px solid #333;
            border-radius: 15px; 
            padding: 40px;
            margin: 20px 50px;
        """)
        diag_layout.addWidget(self.lbl_fault_status)
        diag_layout.addStretch()

        # TAB 2: GATEWAY (F411) VERİLERİ
        self.tab_f411 = pg.GraphicsLayoutWidget()
        self.tabs.addTab(self.tab_f411, "F411 Canlı Veri")

        # Tork ve Hız
        p_trq = self.tab_f411.addPlot(title="Tork (Nm)", row=0, col=0)
        self.curve_trq = p_trq.plot(pen='#FFC107', width=2)
        p_trq.showGrid(x=True, y=True, alpha=0.3)

        p_rpm = self.tab_f411.addPlot(title="Hız (RPM)", row=0, col=1)
        self.curve_rpm = p_rpm.plot(pen='#E040FB', width=2)
        p_rpm.showGrid(x=True, y=True, alpha=0.3)

        self.tab_f411.nextRow()

        # Kısmi FFT
        p_part = self.tab_f411.addPlot(title="AI Giriş Spektrumu (0-160 Hz)", row=1, col=0, colspan=2)
        self.curve_part_fft = p_part.plot(pen='#29B6F6', fillLevel=-140, brush=(41, 182, 246, 30))
        p_part.setXRange(0, 160)
        p_part.setYRange(-20, 80)
        p_part.showGrid(x=True, y=True, alpha=0.3)

        # TAB 3: FULL FFT (F401)
        self.tab_fft = pg.GraphicsLayoutWidget()
        self.tabs.addTab(self.tab_fft, "Full FFT Analiz")
        p1 = self.tab_fft.addPlot(title="Tam Spektrum (0-1000 Hz)")
        self.curve_full_fft = p1.plot(pen='#00E676', fillLevel=-140, brush=(0, 230, 118, 50))
        p1.setLabel('bottom', 'Frekans (Hz)')
        p1.showGrid(x=True, y=True, alpha=0.3)
        p1.setXRange(0, 1000)
        p1.setYRange(-20, 100)

        # TAB 4: SPEKTROGRAM
        self.tab_spec = pg.GraphicsLayoutWidget()
        self.tabs.addTab(self.tab_spec, "Spektrogram")
        p_spec = self.tab_spec.addPlot(title="Zaman-Frekans Analizi")
        self.img_item_spec = pg.ImageItem()
        p_spec.addItem(self.img_item_spec)
        # Ölçekleme
        tr = QtGui.QTransform()
        tr.scale(1, FREQ_RES)
        self.img_item_spec.setTransform(tr)
        # Renk Barı
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img_item_spec)
        hist.gradient.setColorMap(self.colormap)
        hist.setLevels(-120, 20)
        self.tab_spec.addItem(hist)

        # TAB 5: SİMÜLASYON KONTROL
        self.tab_sim = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_sim, "Simülasyon Ayarları")
        self._setup_sim_tab()

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)  # 20 FPS
        self.main_window.show()

    def _setup_sim_tab(self):
        l = QtWidgets.QVBoxLayout(self.tab_sim)
        h = QtWidgets.QHBoxLayout()
        self.combo_files = QtWidgets.QComboBox()
        btn = QtWidgets.QPushButton("Dosyaları Yenile")
        btn.clicked.connect(self.refresh_files)
        h.addWidget(self.combo_files);
        h.addWidget(btn)
        l.addLayout(h)

        h2 = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Simülasyonu Başlat");
        b1.clicked.connect(self.start_sim_worker)
        b2 = QtWidgets.QPushButton("Durdur");
        b2.clicked.connect(self.stop_sim_worker)
        h2.addWidget(b1);
        h2.addWidget(b2)
        l.addLayout(h2)

        self.sim_log = QtWidgets.QTextEdit()
        self.sim_log.setReadOnly(True)
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

    # --- SUNUCU THREADLERİ ---

    def server_thread_f401_fft(self):
        """F401'den gelen Full FFT verisini dinler (Port 5007)"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            log_debug(f"F401 Server Başlatıldı: {PC_IP}:{PORT_RECV_F401_FFT}")
            s.bind((PC_IP, PORT_RECV_F401_FFT))
            s.listen(1)
            while self.running:
                conn, addr = s.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                buf = bytearray()
                while self.running:
                    try:
                        chunk = conn.recv(8192)
                        if not chunk: break
                        buf.extend(chunk)
                        while len(buf) >= FFT_PACKET_SIZE:
                            pkt = buf[:FFT_PACKET_SIZE]
                            buf = buf[FFT_PACKET_SIZE:]
                            arr = np.frombuffer(pkt, dtype='<f4')
                            with self.lock: self.full_iq_fft = arr
                    except:
                        break
                conn.close()
        except Exception as e:
            log_debug(f"F401 Server Hatası: {e}")

    def server_thread_f411_data(self):
        """
        F411'den gelen AI Sonuçlarını ve Verileri dinler (Port 6001).
        Paket Boyutu: 404 (Fast) veya 648 (Slow + AI)
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            log_debug(f"F411 Gateway Server Başlatıldı: {PC_IP}:{PORT_RECV_F411_DATA}")
            s.bind((PC_IP, PORT_RECV_F411_DATA))
            s.listen(1)

            while self.running:
                log_debug("F411 Bağlantısı Bekleniyor...")
                conn, addr = s.accept()
                log_debug(f"F411 BAĞLANDI -> {addr}")

                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                buf = bytearray()

                while self.running:
                    try:
                        chunk = conn.recv(4096)
                        if not chunk:
                            log_debug("F411 Bağlantısı Koptu.")
                            break
                        buf.extend(chunk)

                        while len(buf) >= 4:
                            # Header kontrolü (Buffer'dan silmeden oku)
                            header = struct.unpack('<I', buf[:4])[0]

                            if header == 0x1111AAAA:  # FAST DATA
                                if len(buf) >= FAST_SIZE:  # 404 Byte
                                    pkt = buf[:FAST_SIZE]
                                    buf = buf[FAST_SIZE:]  # Bufferdan düş

                                    data = struct.unpack(FAST_FMT, pkt)
                                    # data[1..50]: Tork, data[51..100]: Hız
                                    with self.lock:
                                        self.trq_history = np.roll(self.trq_history, -50)
                                        self.trq_history[-50:] = np.array(data[1:51])
                                        self.rpm_history = np.roll(self.rpm_history, -50)
                                        self.rpm_history[-50:] = np.array(data[51:101])
                                else:
                                    break  # Yeterli veri yok, bekle

                            elif header == 0x2222BBBB:  # SLOW DATA (FFT + AI Result)
                                if len(buf) >= SLOW_SIZE:  # 648 Byte
                                    pkt = buf[:SLOW_SIZE]
                                    buf = buf[SLOW_SIZE:]  # Bufferdan düş

                                    # Format: Header(I) + 160f + Result(I)
                                    data = struct.unpack(SLOW_FMT, pkt)

                                    # data[0]: Header
                                    # data[1:-1]: FFT verileri (160 adet)
                                    # data[-1]: AI Sonucu (Son eleman)

                                    fft_part = np.array(data[1:-1])
                                    fault_val = data[-1]

                                    with self.lock:
                                        self.f411_fft = fft_part
                                        self.latest_fault_code = fault_val

                                    log_debug(f">>> AI TESPİTİ: Kod {fault_val} -> {FAULT_CODES.get(fault_val, '???')}")
                                else:
                                    break  # Yeterli veri yok, bekle
                            else:
                                # Header eşleşmedi, senkronizasyon bozuldu.
                                # 1 byte kaydırarak tekrar dene.
                                buf = buf[1:]
                    except Exception as e:
                        log_debug(f"Paket İşleme Hatası: {e}")
                        break
                conn.close()
        except Exception as e:
            log_debug(f"F411 Server Init Hatası: {e}")

    def start_system(self):
        if not self.running:
            self.running = True
            # Daemon=True, program kapanınca threadlerin de kapanmasını sağlar
            threading.Thread(target=self.server_thread_f401_fft, daemon=True).start()
            threading.Thread(target=self.server_thread_f411_data, daemon=True).start()
            self.btn_start.setEnabled(False)
            self.btn_start.setText("SUNUCULAR AKTİF - VERİ BEKLENİYOR")

    def update_plots(self):
        d_full, d_trq_hist, d_rpm_hist, d_part, fault_code = None, None, None, None, 0

        with self.lock:
            if self.full_iq_fft is not None: d_full = self.full_iq_fft.copy()
            d_trq_hist = self.trq_history.copy()
            d_rpm_hist = self.rpm_history.copy()
            if self.f411_fft is not None: d_part = self.f411_fft.copy()
            fault_code = self.latest_fault_code

        # 1. Full FFT & Spec
        if d_full is not None:
            try:
                d_full_db = 20 * np.log10(d_full + 1e-9)
                self.curve_full_fft.setData(self.freqs_full, d_full_db)
                self.spec_buf = np.roll(self.spec_buf, -1, axis=0)
                self.spec_buf[-1] = d_full_db
                self.img_item_spec.setImage(self.spec_buf, autoLevels=False)
            except:
                pass

        # 2. Tork/RPM
        x_axis = np.arange(HISTORY_SIZE)
        self.curve_trq.setData(x_axis, d_trq_hist)
        self.curve_rpm.setData(x_axis, d_rpm_hist)

        # 3. Kısmi FFT (AI Girişi)
        if d_part is not None:
            try:
                d_part_db = 20 * np.log10(d_part + 1e-9)
                self.curve_part_fft.setData(self.freqs_partial, d_part_db)
            except:
                pass

        # 4. AI ARIZA GÖSTERGESİ GÜNCELLEME
        txt = FAULT_CODES.get(fault_code, f"BİLİNMEYEN KOD ({fault_code})")
        color = FAULT_COLORS.get(fault_code, "#333")

        self.lbl_fault_status.setText(txt)
        # Yanıp sönme efekti eklenebilir veya sadece renk değişimi
        self.lbl_fault_status.setStyleSheet(f"""
            font-size: 56px; 
            font-weight: bold; 
            color: #FFF; 
            background-color: {color}; 
            border: 2px solid #FFF;
            border-radius: 15px; 
            padding: 40px;
            margin: 20px 50px;
        """)


if __name__ == "__main__":
    app = MotorMonitor()
    QtWidgets.QApplication.instance().exec()