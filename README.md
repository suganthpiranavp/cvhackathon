# ⚡ VisionOps: Person Detection, Tracking & Visual Re-Identification Engine

A lightning-fast, production-grade Person Detection, Tracking, and Visual Re-Identification system built in Python with a premium Streamlit dark-mode dashboard.

---

## 🚀 Key Features

- **Native Ultra-Fast Stack**: Powered by `ultralytics.YOLO` (YOLOv8 Nano) with built-in BoT-SORT (`botsort.yaml`) motion tracking and appearance Re-ID feature embeddings.
- **Class Filtering at Inference**: Strictly filters inference to human targets (`classes=0`) to eliminate unnecessary latency.
- **Synchronized Real-Time Live Streaming**: During inference, renders the raw decoded stream and the AI-tracked stream in 1:1 frame synchronicity.
- **Dual Interactive Playback Matrix**: Once processing completes, both the source clip and the AI-tracked video (with burnt-in neon boxes and persistent IDs) are mounted in native browser video players side-by-side with full scrub, pause, and seek controls.
- **Live Telemetry & Metrics**: Real-time status cards tracking:
  - Total Unique People identified in session
  - Active Targets currently in frame
  - Engine Speed (FPS) & processing latency
- **Dynamic Boundary Event Logs**: Live table logging target entry (`🟢 ENTERED BOUNDARY`) and exit (`🔴 EXITED BOUNDARY`) transitions with exact timestamps and frame indices.
- **Web-Safe H.264 Transmutation**: Transmutes raw OpenCV `mp4v` video output into browser-safe H.264 via FFmpeg.
- **One-Click Download**: Direct high-visibility download button to export the analyzed tracking video.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suganthpiranavp/cvhackathon.git
   cd cvhackathon
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard:**
   ```bash
   python3 -m streamlit run app.py
   ```

---

## 📦 Tech Stack

- **Computer Vision & Inference**: OpenCV (`cv2`), Ultralytics YOLOv8, PyTorch
- **Tracking & Re-ID**: BoT-SORT (`botsort.yaml`), `lapx`
- **Dashboard UI**: Streamlit (Enterprise Dark Theme)
- **Container Transcoding**: FFmpeg (`libx264` / `imageio-ffmpeg`)
- **Data & Telemetry**: Pandas, NumPy
