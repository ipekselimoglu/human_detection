"""
Webcam Human Detection System using YOLOv8
------------------------------------------

Bu program, bilgisayar kamerasından (veya belirtilen bir video kaynağından) 
gelen canlı görüntüde **insan tespiti** yapar.

Ultralytics YOLOv8 modelini kullanarak her karedeki nesneleri analiz eder ve
insan (class 0) tespit edildiğinde ekrana kırmızı renkle **"HUMAN DETECTED"**,
insan bulunmadığında ise yeşil renkle **"No Human Detected"** mesajını yazar.

Ek olarak, insan tespiti devam ettiği sürece sistem:
- Her 5 saniyede bir anlık görüntüyü (frame) `alerts/webcam_frames/` klasörüne kaydeder.
- Fotoğraflar tarih ve saat bilgisini içerir (Windows uyumlu dosya isimleriyle).

Kullanım:
    - Kamerayı açmak için programı çalıştırın.
    - İzleme esnasında görüntü penceresinde gerçek zamanlı sonuçları görebilirsiniz.
    - Programdan çıkmak için klavyeden **Q** tuşuna basın.

Bağımlılıklar:
    - ultralytics (YOLOv8)
    - opencv-python (cv2)
    - Python 3.8+
"""

from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime

# ======= AYARLAR =======
VIDEO_SOURCE = 0                   # 0 = varsayılan kamera
MODEL_NAME = "yolov8n.pt"          # Model
CONF_THRESHOLD = 0.35              # Güven eşiği (biraz düşürüldü, daha hassas)
SAVE_INTERVAL_SEC = 5              # Her 5 saniyede bir kayıt
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ========================

# Modeli yükle
model = YOLO(MODEL_NAME)

# Klasör oluştur
os.makedirs("alerts/webcam_frames", exist_ok=True)

alert_id = 0
last_save_time = time.time()

# Kamera başlat
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🚀 Webcam monitoring started (press Q to quit)\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Kamera akışı okunamadı, tekrar deneniyor...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        continue

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # YOLO tespiti
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    # 🔎 İnsan sınıfı (class 0) + güven eşiği kontrolü
    persons = [b for b in boxes if int(b.cls[0]) == 0 and b.conf[0] > CONF_THRESHOLD]
    has_person = len(persons) > 0

    if has_person:
        # 🔴 “Human Detected” yazısı
        cv2.putText(frame, "HUMAN DETECTED", (20, 60), FONT, 1.8, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, timestamp, (20, frame.shape[0] - 20),
                    FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # 📸 Gerçek zaman kontrollü kayıt
        current_time = time.time()
        if current_time - last_save_time >= SAVE_INTERVAL_SEC:
            alert_id += 1
            last_save_time = current_time
            save_path = f"alerts/webcam_frames/human_{alert_id}_{timestamp.replace(':', '-')}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"📸 İnsan tespit edildi → {save_path}")

    else:
        # 🟢 “No Human Detected” yazısı
        cv2.putText(frame, "No Human Detected", (20, 60), FONT, 1.8, (0, 255, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, timestamp, (20, frame.shape[0] - 20),
                    FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Canlı görüntü
    cv2.imshow("Webcam Monitor", frame)

    # Çıkış için Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n🛑 İzleme sonlandırıldı.")
        break

cap.release()
cv2.destroyAllWindows()
