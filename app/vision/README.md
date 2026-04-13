# FocusGuardian - Vision Module

## Cấu trúc

```
app/vision/
├── __init__.py          # Module exports
├── camera.py            # Webcam capture với threading
├── face_mesh.py         # MediaPipe Face Mesh wrapper
├── head_pose.py         # Head pose estimation (solvePnP)
├── blink.py             # Blink detection (EAR)
├── hands.py             # Hand detection & write score
├── phone_detector.py    # Phone detection (stub/YOLOv8)
└── demo.py              # Demo script test tất cả modules
```

## Cài đặt

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## Chạy Demo

```bash
# Từ thư mục gốc App
python -m app.vision.demo
```

### Điều khiển Demo:
- `q` - Thoát
- `c` - Calibrate (nhìn thẳng vào màn hình)
- `r` - Reset tracking
- `h` - Bật/tắt hand detection
- `d` - Bật/tắt debug overlays

## Các Module

### 1. Camera (`camera.py`)
- Capture webcam với thread riêng
- Tự động resize frame cho processing
- FPS counter và error handling

### 2. Face Mesh (`face_mesh.py`)
- Wrapper cho MediaPipe Face Mesh
- Trả về 478 landmarks (bao gồm iris)
- Helper functions cho pixel/normalized coords

### 3. Head Pose (`head_pose.py`)
- Sử dụng solvePnP từ 10 landmarks
- Output: pitch (cúi/ngẩng), yaw (trái/phải), roll (nghiêng)
- Smoothing với EMA
- Calibration support

**Thresholds quan trọng:**
- `head_down_threshold`: -15° (pitch < -15 = cúi xuống)
- `look_away_threshold`: 30° (|yaw| > 30 = nhìn đi)

### 4. Blink Detection (`blink.py`)
- Eye Aspect Ratio (EAR) calculation
- Blink counting và rate
- PERCLOS cho drowsiness detection

**Thresholds quan trọng:**
- `ear_threshold`: 0.21 (EAR < 0.21 = mắt nhắm)
- `perclos_threshold`: 0.15 (15% mắt nhắm = buồn ngủ)

### 5. Hand Analyzer (`hands.py`)
- Phát hiện tay với MediaPipe Hands
- Phân vùng: upper/middle/lower
- **hand_write_score**: 0-1, score cao = đang viết/ghi chép

**Logic phát hiện viết:**
```
write_score = 0.4 * region_score      # Tay ở vùng lower
            + 0.4 * motion_score      # Motion nhỏ, đều
            + 0.2 * stability_score   # Vị trí ổn định
```

### 6. Phone Detector (`phone_detector.py`)
- **Mặc định**: Stub mode (không phát hiện)
- **Optional**: YOLOv8n cho detection thật

Để bật phone detection:
```python
from app.vision.phone_detector import PhoneDetector, PhoneDetectorConfig

detector = PhoneDetector(PhoneDetectorConfig(
    enabled=True,
    model_type="yolov8",
    confidence_threshold=0.5
))
```

## Điều chỉnh Threshold cho "Cúi làm bài"

### Vấn đề
Khi học sinh cúi xuống làm bài:
- Head pitch giảm (âm nhiều)
- Nhưng vẫn đang tập trung

### Giải pháp: Triangulation
Không chỉ dựa vào head pitch mà kết hợp:

1. **Hand write score cao** (> 0.5)
   - Tay ở vùng lower
   - Motion nhỏ, đều đặn

2. **Glances up định kỳ**
   - Học sinh thường ngẩng lên nhìn màn hình/sách rồi cúi xuống viết

3. **Không có phone detected**

### Cách tune:
```python
# Trong hands.py - HandConfig
lower_region_threshold: float = 0.55   # Giảm nếu camera cao
write_score_threshold: float = 0.5     # Giảm để dễ detect writing hơn
write_motion_min: float = 0.05         # Motion tối thiểu
write_motion_max: float = 0.4          # Motion tối đa (quá nhiều = không viết)

# Trong head_pose.py - HeadPoseConfig
head_down_threshold: float = -15.0     # Tăng (ít âm hơn) nếu muốn strict hơn
```

## Performance Tips

1. **Resize frame** trước khi process (480x360 thay vì 640x480)
2. **Giảm FPS** nếu cần (15-20 FPS đủ cho use case này)
3. **Tắt hand detection** nếu không cần
4. **Chạy riêng thread** cho mỗi detector (trong production)
