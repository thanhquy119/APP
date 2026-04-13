# 🎯 FocusGuardian

**Ứng dụng desktop theo dõi tập trung học tập sử dụng webcam và AI**

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)
![MediaPipe](https://img.shields.io/badge/AI-MediaPipe_Tasks-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 Giới thiệu

FocusGuardian là ứng dụng desktop giúp theo dõi và cải thiện khả năng tập trung khi học tập hoặc làm việc. Sử dụng webcam và các mô hình AI, ứng dụng có thể:

- 👁️ **Phát hiện trạng thái tập trung** - Nhận biết khi bạn đang đọc màn hình
- ✍️ **Phân biệt ghi chép** - Phân biệt cúi đầu ghi chép vs mất tập trung
- 📱 **Phát hiện mất tập trung** - Cảnh báo khi sử dụng điện thoại
- 😴 **Phát hiện mệt mỏi** - Nhận biết dấu hiệu buồn ngủ qua chớp mắt
- ⏰ **Nhắc nghỉ giải lao** - Pomodoro technique tích hợp
- 🎮 **Mini games** - Trò chơi nhỏ để thư giãn khi nghỉ

## 🚀 Tính năng chính

### Phát hiện 6 trạng thái tập trung

| Trạng thái | Mô tả | Màu sắc |
|------------|-------|---------|
| 👁️ ON_SCREEN_READING | Đang đọc màn hình, tập trung | 🟢 Xanh lá |
| ✍️ OFFSCREEN_WRITING | Đang ghi chép (vẫn tập trung) | 🔵 Xanh dương |
| 📱 PHONE_DISTRACTION | Sử dụng điện thoại | 🟠 Cam đậm |
| 😴 DROWSY_FATIGUE | Mệt mỏi, buồn ngủ | 🟡 Vàng cam |
| 🚶 AWAY | Không có mặt | ⚪ Xám |
| ❓ UNCERTAIN | Không xác định được | 🔘 Xám xanh |

### Thuật toán thông minh

- **MediaPipe Tasks API**: Sử dụng API mới nhất, tương thích Python 3.12/3.13+
- **FaceLandmarker**: 478 điểm 3D landmarks + blendshapes cho phát hiện khuôn mặt chính xác
- **HandLandmarker**: 21 điểm 3D landmarks cho phân tích tay và phát hiện viết
- **Temporal Windows**: Phân tích trong cửa sổ thời gian 10s và 30s
- **Hysteresis**: Tránh chuyển trạng thái liên tục
- **Head Pose via solvePnP**: Ước lượng góc pitch/yaw/roll của đầu

## 📦 Cài đặt

### Yêu cầu hệ thống

- Windows 10/11 (64-bit)
- **Python 3.12 hoặc 3.13** (khuyến nghị)
- Webcam
- RAM: tối thiểu 4GB (khuyến nghị 8GB)
- Internet: Cần để tải model files lần đầu (~15MB)

### Cài đặt từ source

```bash
# Clone repository
git clone https://github.com/yourusername/focusguardian.git
cd focusguardian

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Tải model files (tự động khi chạy lần đầu, hoặc chạy thủ công)
python -m app.vision.model_manager

# Chạy ứng dụng
python main.py
```

### Cài đặt từ release (Windows)

1. Tải file `FocusGuardian-Setup.exe` từ [Releases](https://github.com/yourusername/focusguardian/releases)
2. Chạy installer và làm theo hướng dẫn
3. Mở FocusGuardian từ Start Menu

## 🎮 Hướng dẫn sử dụng

### Bắt đầu nhanh

1. **Khởi động ứng dụng** - Chạy `python main.py` hoặc mở từ shortcut
2. **Cho phép camera** - Ứng dụng sẽ yêu cầu quyền truy cập webcam
3. **Nhấn "Bắt đầu"** - Camera sẽ bật và bắt đầu theo dõi
4. **Học tập bình thường** - Ứng dụng tự động theo dõi và ghi nhận

### Cài đặt

Nhấn nút ⚙️ **Cài đặt** để tùy chỉnh:

- **Camera**: Chọn webcam, độ phân giải
- **Tập trung**: Điều chỉnh ngưỡng phát hiện
- **Nghỉ giải lao**: Thời gian làm việc/nghỉ, Pomodoro
- **Thông báo**: Bật/tắt các loại thông báo

### Mini Games

Khi nghỉ giải lao, bạn có thể chơi:

- 🌬️ **Bài tập thở**: Hít thở 4-4-6 giúp thư giãn
- 🧠 **Ghép thẻ nhớ**: Luyện trí nhớ ngắn hạn
- ⌨️ **Luyện gõ**: Kiểm tra tốc độ đánh máy

## 🏗️ Cấu trúc dự án

```
App/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── config.json            # User configuration (auto-generated)
├── app.spec               # PyInstaller spec file
│
├── app/
│   ├── __init__.py
│   │
│   ├── vision/            # Computer Vision modules
│   │   ├── camera.py      # Camera capture
│   │   ├── vision_pipeline.py # Unified vision processing (NEW)
│   │   ├── face_landmarker.py # MediaPipe Tasks FaceLandmarker (NEW)
│   │   ├── hand_landmarker.py # MediaPipe Tasks HandLandmarker (NEW)
│   │   ├── model_manager.py   # Model download & cache (NEW)
│   │   ├── camera.py          # Camera capture
│   │   ├── face_mesh.py       # (Legacy - deprecated)
│   │   ├── head_pose.py       # (Legacy - deprecated)
│   │   ├── blink.py           # (Legacy - deprecated)
│   │   ├── hands.py           # (Legacy - deprecated)
│   │   └── phone_detector.py  # Phone detection
│   │
│   ├── logic/             # Core logic
│   │   ├── focus_engine.py    # State machine
│   │   └── __init__.py
│   │
│   ├── utils/             # Utilities
│   │   ├── ring_buffer.py     # Time-windowed buffer
│   │   ├── win_idle.py        # Windows idle detection
│   │   └── __init__.py
│   │
│   └── ui/                # PyQt6 UI
│       ├── main_window.py     # Main window
│       ├── settings_dialog.py # Settings
│       ├── mini_games.py      # Break games
│       ├── tray.py            # System tray
│       └── __init__.py
│
├── assets/
│   ├── models/            # MediaPipe model files (auto-downloaded)
│   │   ├── face_landmarker.task
│   │   └── hand_landmarker.task
│   └── nature/            # Nature sounds for relaxation
│
└── tests/
    └── test_focus_engine.py   # Unit tests
```

## 🔧 Build Windows EXE

### Bước 1: Chuẩn bị

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Tải model files trước (quan trọng!)
python -m app.vision.model_manager

# Kiểm tra models đã tải
dir assets\models
# Phải có: face_landmarker.task, hand_landmarker.task
```

### Bước 2: Build với PyInstaller

```bash
# Build
pyinstaller app.spec

# Output sẽ ở thư mục dist/FocusGuardian/
```

### Bước 3: Test EXE

```bash
# Chạy thử
dist\FocusGuardian\FocusGuardian.exe
```

### Tạo Installer (tùy chọn)

Sử dụng [Inno Setup](https://jrsoftware.org/isinfo.php) hoặc [NSIS](https://nsis.sourceforge.io/) để tạo installer.

## 🧪 Testing

```bash
# Chạy tất cả tests
python -m pytest tests/ -v

# Chạy tests với coverage
python -m pytest tests/ --cov=app --cov-report=html
```

## 📊 Thuật toán Focus Engine

### State Machine

```
                    ┌─────────────────────────────────────┐
                    │           UNCERTAIN                 │
                    │      (Initial / Can't detect)       │
                    └────────────────┬────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │ ON_SCREEN_      │    │ OFFSCREEN_      │    │     AWAY        │
   │ READING         │◄──►│ WRITING         │    │ (No face)       │
   │ (Looking at     │    │ (Head down +    │    └─────────────────┘
   │  screen)        │    │  writing)       │
   └────────┬────────┘    └────────┬────────┘
            │                      │
            ▼                      ▼
   ┌─────────────────┐    ┌─────────────────┐
   │ PHONE_          │    │ DROWSY_         │
   │ DISTRACTION     │    │ FATIGUE         │
   │ (Phone/head     │    │ (Eyes closed/   │
   │  down no write) │    │  low EAR)       │
   └─────────────────┘    └─────────────────┘
```

### Công thức chính

**Head Down Detection:**
```
head_down = (pitch < HEAD_DOWN_THRESHOLD)  # Default: -15°
```

Bạn có thể tinh chỉnh trực tiếp trong Cài đặt > Tập trung:

- `Cúi đầu (pitch)` (`head_down_threshold`)
- `Ngưỡng mắt nhìn xuống` (`eye_look_down_threshold`)
- `Cúi + mắt xuống để tính phone` (`phone_eye_down_min_duration`, mặc định 45 giây)
- `Chớp thấp (vẫn tập trung)` (`blink_rate_low_screen_max`)
- `Chớp cao (mệt mỏi)` (`blink_rate_high_fatigue_min`)

**Writing Detection (CRITICAL):**
```
is_writing = head_down AND (hand_write_score > 0.4) AND hand_present
```

**Phone Distraction (head-down disambiguation):**
```
is_phone = (
    head_down
    AND eye_look_down_sustained >= 45s
    AND eye_down_ratio >= threshold
    AND low_writing_evidence
)
```

**Head-down but still focused on screen:**
```
is_still_focused = head_down AND eye_look_up_high AND blink_rate_low
```

**Head-down + high blink-rate:**
```
is_fatigue_like = head_down_sustained AND blink_rate_high
```

**Drowsy Detection:**
```
is_drowsy = (eye_closure_ratio > 0.3) OR (avg_EAR < 0.18) OR (idle_long AND head_down)
```

### Cơ sở khoa học dùng để đặt ngưỡng

- Stern, Walrath, Goldstein (1984): tần suất chớp mắt tự phát khi nghỉ thường quanh mức 15-20 lần/phút.
- Patel et al. (1991, Optometry and Vision Science): khi làm việc với màn hình, tần suất chớp mắt giảm đáng kể (thường thấp hơn trạng thái nghỉ).
- Tsubota & Nakamori (1993, NEJM): công việc màn hình liên quan giảm chớp mắt và thay đổi ổn định màng phim nước mắt.

Các ngưỡng trong ứng dụng được thiết kế theo hướng bảo thủ từ các mốc trên và cho phép tinh chỉnh theo từng người dùng/camera.

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 🙏 Credits

- [MediaPipe](https://mediapipe.dev/) - Face Mesh và Hand Tracking
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI Framework
- [OpenCV](https://opencv.org/) - Computer Vision

---

**Made with ❤️ for students and knowledge workers**
