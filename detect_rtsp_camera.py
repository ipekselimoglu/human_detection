"""
RTSP Tabanlı Gerçek Zamanlı İnsan Tespiti (YOLOv8)
===================================================

Bu betik, RTSP (Real-Time Streaming Protocol) üzerinden yayın yapan bir IP kameradan
görüntü alarak YOLOv8 nesne algılama modeli ile gerçek zamanlı olarak insan tespiti yapar.

Program, RTSP bağlantısı üzerinden kareleri (frame) sürekli olarak okur, her kare üzerinde
YOLOv8 modelini çalıştırır ve "person" sınıfı (class ID: 0) tespit edildiğinde
belirli aralıklarla kareleri kaydeder. Aynı zamanda ekrana tespit durumu ve zaman bilgisini yazar.

Ana Özellikler
--------------
- RTSP üzerinden canlı yayın izleme ve gerçek zamanlı tespit
- YOLOv8 derin öğrenme modeli ile insan algılama
- RTSP bağlantısı koptuğunda otomatik yeniden bağlanma
- Belirlenen saniye aralığında tespit edilen kareleri kaydetme
- Görüntü üzerine bilgi metni ve zaman damgası ekleme

Parametreler
------------
RTSP_URL : str
    IP kameranın RTSP bağlantı adresi (örn. "rtsp://kullanici:parola@192.168.1.10:554/stream1")
MODEL_NAME : str
    Kullanılacak YOLOv8 model dosyasının adı veya yolu (örn. "yolov8n.pt")
CONF_THRESHOLD : float
    Modelin tahminleri için minimum güven eşiği (0.0 – 1.0 arası)
SAVE_INTERVAL_SEC : int
    İnsan tespit edildiğinde karelerin kaç saniyede bir kaydedileceği
RECONNECT_DELAY : int
    RTSP bağlantısı koptuğunda yeniden denemeden önce beklenecek süre (saniye)
FONT : int
    OpenCV yazı tipi (ekranda bilgi metinleri için kullanılır)

Fonksiyonlar
------------
connect_rtsp(url: str) -> cv2.VideoCapture | None
    RTSP bağlantısı kurar. Bağlantı başarılıysa cv2.VideoCapture nesnesi döner,
    başarısız olursa None döner.

Çalışma Akışı
-------------
1. YOLOv8 modeli belirtilen dosyadan yüklenir.
2. RTSP bağlantısı `connect_rtsp()` fonksiyonu ile kurulur.
3. Akıştan alınan kareler üzerinde sürekli olarak model çalıştırılır.
4. Model çıktılarında kişi (class ID = 0) tespit edilirse:
       - Ekrana “Human Detected” yazılır.
       - Kare üzerine zaman bilgisi eklenir.
       - Belirlenen aralıkta kare diske kaydedilir.
5. Bağlantı kesilirse sistem otomatik olarak yeniden bağlanmayı dener.
6. Kullanıcı “Q” tuşuna bastığında program sonlandırılır.

Bağımlılıklar
--------------
- ultralytics
- opencv-python
- os
- datetime
- time

Çıktılar
--------
- Terminal üzerinde tespit ve bağlantı durumu mesajları
- Kaydedilen görüntü kareleri: alerts/rtsp_frames/ klasöründe tutulur
"""

from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import time

# ======= AYARLAR =======
RTSP_URL = "rtsp://kullanici:parola@192.168.1.10:554/stream1"  # Kameranın RTSP bağlantı linki
MODEL_NAME = "yolov8n.pt"       # YOLOv8 model dosyası
CONF_THRESHOLD = 0.45           # Minimum güven eşiği
SAVE_INTERVAL_SEC = 5           # Her 5 saniyede bir kayıt
FONT = cv2.FONT_HERSHEY_SIMPLEX # Yazı tipi
RECONNECT_DELAY = 5             # RTSP koparsa 5 sn sonra yeniden dene
# ========================

# Modeli yükle
model = YOLO(MODEL_NAME)
os.makedirs("alerts/rtsp_frames", exist_ok=True)

def connect_rtsp(url):
    """Kamera bağlantısı kurar, başarısız olursa None döner"""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("RTSP bağlantısı kurulamadı. Tekrar denenecek...")
        return None
    print("RTSP bağlantısı başarılı.")
    return cap

cap = connect_rtsp(RTSP_URL)
alert_id = 0
frame_count = 0
last_save_time = time.time() - SAVE_INTERVAL_SEC

print("Real-time RTSP human detection started... (Kapatmak için 'Q' tuşuna basın)")

while True:
    if cap is None:
        time.sleep(RECONNECT_DELAY)
        cap = connect_rtsp(RTSP_URL)
        continue

    ret, frame = cap.read()
    if not ret:
        print("Akış kesildi. Yeniden bağlanılıyor...")
        cap.release()
        cap = None
        continue

    frame_count += 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # YOLO tahmini
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    # İnsan tespiti
    if boxes is not None and len(boxes) > 0:
        cls = boxes.cls.cpu().numpy().astype(int)
        has_person = any(c == 0 for c in cls)
    else:
        has_person = False

    if has_person:
        cv2.putText(frame, "Human Detected", (25, 50), FONT, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, timestamp, (25, frame.shape[0] - 25), FONT, 0.9, (255, 255, 255), 2)

        # Zaman aralığı kontrolü
        if time.time() - last_save_time >= SAVE_INTERVAL_SEC:
            alert_id += 1
            last_save_time = time.time()
            save_path = f"alerts/rtsp_frames/human_{alert_id}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"İnsan tespit edildi → {save_path}")
    else:
        cv2.putText(frame, "No Human Detected", (25, 50), FONT, 1, (0, 255, 0), 2)

    # Görüntü göster
    cv2.imshow("RTSP Human Detection", frame)

    # Çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nİzleme sonlandırıldı. Toplam {alert_id} kare kaydedildi.")




