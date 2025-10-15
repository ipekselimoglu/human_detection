"""
Human Detection from Video using YOLOv8
---------------------------------------

Bu program, belirtilen bir video dosyasında (örneğin `input8.mp4`) 
**YOLOv8 derin öğrenme modeli** kullanarak insan tespiti yapar.

Program, videoyu kare kare (frame by frame) analiz eder ve:
- Her karede YOLOv8 modelini çalıştırarak nesne sınıflarını belirler.
- Eğer karede insan (class 0) tespit edilirse ekrana bilgi yazar.
- İnsan tespiti devam ederken, her 5 saniyede bir (ayar `SAVE_INTERVAL_SEC`)
  tespit edilen kareyi zaman damgası ile birlikte `alerts/frames/` klasörüne kaydeder.

Özellikler:
    • YOLOv8 modeli ile yüksek doğrulukta insan tespiti.
    • Her 5 saniyede bir kare kaydı (fps’e göre otomatik hesaplanır).
    • Ekran üzerine “Human Detected” ve tarih/saat bilgisi eklenir.
    • İnsan bulunmayan kareler terminalde “İnsan tespit edilmedi.” olarak raporlanır.

Parametreler:
    VIDEO_PATH (str): Analiz edilecek video dosyasının yolu.
    MODEL_NAME (str): Kullanılacak YOLOv8 model dosyası (örn. 'yolov8n.pt').
    CONF_THRESHOLD (float): Minimum güven eşiği (%0.0–1.0 arası).
    SAVE_INTERVAL_SEC (int): İnsan bulunduğunda kaç saniyede bir kare kaydedileceği.
    FONT: Ekrana yazı yazmak için kullanılacak OpenCV fontu.

Çıktılar:
    - `alerts/frames/` klasörüne kaydedilen JPEG formatında tespit kareleri.
    - Terminal üzerinde tespit logları (frame numarası ve zaman bilgisi).
    - İşlem sonunda toplam kaç kare kaydedildiği bilgisi.

Kullanım:
    1. `VIDEO_PATH` değişkenine analiz edilecek video dosyasını girin.
    2. Programı çalıştırın.
    3. Çalışma tamamlandığında tespit edilen kareler `alerts/frames/` klasöründe bulunur.

Bağımlılıklar:
    - ultralytics
    - opencv-python
    - Python 3.8 veya üzeri
"""


from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# ======= AYARLAR =======
VIDEO_PATH = "sample/input8.mp4"          # Analiz edilecek video
MODEL_NAME = "yolov8n.pt"          # Küçük ve hızlı model
CONF_THRESHOLD = 0.45              # Güven eşiği
SAVE_INTERVAL_SEC = 5             # İnsan varken her 5 saniyede bir kayıt
FONT = cv2.FONT_HERSHEY_SIMPLEX    # Yazı tipi
# ========================

# Modeli yükle
model = YOLO(MODEL_NAME)

# Çıktı klasörü
os.makedirs("alerts/video_frames", exist_ok=True)

# Video aç
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
interval_frames = int(fps * SAVE_INTERVAL_SEC)

alert_id = 0
frame_count = 0
last_save_frame = -interval_frames  # başta -5sn geriden başlasın

print(f"Human detection every {SAVE_INTERVAL_SEC}s started...")

# Frame frame ilerle
for result in model.track(
    source=VIDEO_PATH,
    conf=CONF_THRESHOLD,
    stream=True,
    verbose=False,
    persist=True
):
    frame = result.orig_img
    frame_count += 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Eğer insan yoksa (boxes boşsa veya class != 0)
    if not hasattr(result, "boxes") or len(result.boxes) == 0:
        print(f" Frame {frame_count}: İnsan tespit edilmedi.")
        continue

    cls = result.boxes.cls.cpu().numpy().astype(int)
    has_person = any(c == 0 for c in cls)

    if has_person and (frame_count - last_save_frame) >= interval_frames:
        alert_id += 1
        last_save_frame = frame_count

        print(f" İnsan tespit edildi (Frame {frame_count}) → Kayıt alındı.")

        # Yazılar ekle
        cv2.putText(frame, "Human Detected", (25, 50), FONT, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, timestamp, (25, frame.shape[0] - 25), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Görsel kaydet
        save_path = f"alerts/video_frames/human_{alert_id}.jpg"
        cv2.imwrite(save_path, frame)

cap.release()
print(f"\n İşlem tamamlandı. Toplam {alert_id} kare kaydedildi.")

