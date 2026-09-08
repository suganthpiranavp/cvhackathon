"""
================================================================================
PRODUCTION-GRADE PERSON DETECTION, TRACKING & VISUAL RE-IDENTIFICATION SYSTEM
PHASE 1, 2 & 3: DUAL WORKSPACE STREAMING & POST-PROCESSING PLAYBACK MATRIX
================================================================================
Engineered for ultra-low latency, native BoT-SORT appearance Re-ID, dark-mode
enterprise telemetry, synchronized live streaming, and side-by-side interactive
browser video playback.
"""

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# ENGINE CONFIGURATION CONSTANTS & THEMING
# -----------------------------------------------------------------------------
DEFAULT_MODEL_WEIGHTS = "yolov8n.pt"
TRACKER_CONFIG = "botsort.yaml"  # Native BoT-SORT appearance Re-ID tracker
SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi"]

# Industrial SOC Palette (BGR for OpenCV, Hex for CSS)
CYAN_ACCENT_BGR = (235, 165, 14)       # #0ea5e9 (Cyber Cyan / Command Blue)
CYAN_ACCENT_HEX = "#0ea5e9"
CYAN_LIGHT_HEX = "#38bdf8"
EMERALD_ACCENT_BGR = (129, 185, 16)    # #10b981 (Precision Emerald)
EMERALD_ACCENT_HEX = "#10b981"
CORAL_ACCENT_BGR = (68, 68, 239)       # #ef4444 (Crimson Alert)
CORAL_ACCENT_HEX = "#ef4444"
DARK_HUD_BG_BGR = (26, 24, 15)         # #0f172a (Deep Slate 900)
TEXT_WHITE_BGR = (248, 250, 252)       # #f8fafc (Crisp Off-White)

# Compatibility aliases
NEON_GREEN_BGR = CYAN_ACCENT_BGR
NEON_GREEN_HEX = CYAN_ACCENT_HEX
DARK_BADGE_BG_BGR = DARK_HUD_BG_BGR


@dataclass(frozen=True)
class VideoMetadata:
    """Core video parameters extracted directly from OpenCV hardware demuxer."""
    total_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float


# -----------------------------------------------------------------------------
# 1. HIGH-EFFICIENCY MODEL CACHING (WARM IN-RAM ENGINE)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="⚡ Initializing YOLOv8 Neural Engine into RAM...")
def load_yolo_model(
    openvino_path: str = "yolov8n_openvino_model",
    fallback_weights: str = DEFAULT_MODEL_WEIGHTS,
) -> YOLO:
    """
    Loads OpenVINO compiled model directory for accelerated CPU inference if available;
    otherwise falls back to standard PyTorch weights.
    Streamlit's @st.cache_resource prevents redundant instantiation across UI reruns.
    """
    if os.path.exists(openvino_path):
        model = YOLO(openvino_path)
    else:
        model = YOLO(fallback_weights)
    return model


# -----------------------------------------------------------------------------
# 2. STREAMLINED FILE BUFFERING & METADATA EXTRACTION
# -----------------------------------------------------------------------------
def buffer_uploaded_video(uploaded_file) -> Tuple[str, VideoMetadata]:
    """
    Buffers an uploaded Streamlit video stream to a local temporary file with zero
    data loss, allowing OpenCV (cv2.VideoCapture) to interface directly with the byte stream.
    """
    file_suffix = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise RuntimeError(f"OpenCV failed to open buffered video stream at: {temp_video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # Fallback for variable/unreadable frame rates
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = total_frames / fps if fps > 0 else 0.0

        metadata = VideoMetadata(
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
        )
    finally:
        cap.release()

    return temp_video_path, metadata


# -----------------------------------------------------------------------------
# 3. FAST FFMPEG STREAM RESOLVER & TRANSMUTATION WORKAROUND
# -----------------------------------------------------------------------------
def get_ffmpeg_binary() -> str:
    """Resolves system ffmpeg binary or falls back to bundled imageio-ffmpeg executable."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def convert_to_web_safe_h264(raw_input_path: str, web_output_path: str) -> bool:
    """
    Transmutes raw OpenCV mp4v video containers into web-safe H.264 / AAC MP4 format.
    Guarantees native hardware playback and seeking across all standard web browsers.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    command = f'"{ffmpeg_bin}" -y -i "{raw_input_path}" -vcodec libx264 -pix_fmt yuv420p -f mp4 "{web_output_path}"'
    exit_code = os.system(command)
    return exit_code == 0 and os.path.exists(web_output_path) and os.path.getsize(web_output_path) > 0


# -----------------------------------------------------------------------------
# 4. CUSTOM ENTERPRISE CSS INJECTION
# -----------------------------------------------------------------------------
def apply_enterprise_theme():
    """Injects industrial Security Operations Center (SOC) dark-mode styling."""
    st.markdown(
        """
        <style>
        /* Base Enterprise Canvas (Slate 900 / Deep Obsidian) */
        .stApp {
            background-color: #0b1120;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid #1e293b !important;
        }

        /* Telemetry KPI Cards */
        .metric-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 1.15rem 1.35rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
            transition: all 0.25s ease;
            position: relative;
        }
        .metric-card:hover {
            border-color: #475569;
            box-shadow: 0 6px 22px rgba(14, 165, 233, 0.15);
        }
        .metric-label {
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 0.4rem;
        }
        .metric-value {
            font-size: 2.25rem;
            font-weight: 900;
            line-height: 1.1;
            letter-spacing: -0.02em;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }
        .metric-subtext {
            font-size: 0.76rem;
            color: #64748b;
            margin-top: 0.35rem;
            font-weight: 600;
        }

        /* Stream Header Badges */
        .stream-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.85rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }
        .badge-raw {
            background: rgba(148, 163, 184, 0.12);
            color: #94a3b8;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }
        .badge-live {
            background: rgba(14, 165, 233, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(14, 165, 233, 0.35);
        }
        .badge-playback {
            background: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        /* Action Buttons */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(90deg, #0ea5e9 0%, #0284c7 100%) !important;
            color: #ffffff !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.03em !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            border-radius: 8px !important;
            padding: 0.65rem 1.6rem !important;
            box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3) !important;
            transition: all 0.2s ease-in-out !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: linear-gradient(90deg, #38bdf8 0%, #0ea5e9 100%) !important;
            box-shadow: 0 0 20px rgba(14, 165, 233, 0.5) !important;
            transform: translateY(-1px);
        }

        div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(90deg, #059669 0%, #047857 100%) !important;
            color: #ffffff !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.03em !important;
            border: 1px solid rgba(52, 211, 153, 0.3) !important;
            border-radius: 8px !important;
            padding: 0.75rem 1.8rem !important;
            box-shadow: 0 4px 14px rgba(5, 150, 105, 0.3) !important;
            transition: all 0.2s ease-in-out !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.5) !important;
            transform: translateY(-1px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(placeholder, label: str, value: str, subtext: str, value_color: str = CYAN_LIGHT_HEX):
    """Renders a stylized telemetry card inside a Streamlit placeholder with SOC styling."""
    placeholder.markdown(
        f"""
        <div class="metric-card" style="border-top: 2px solid {value_color};">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color: {value_color};">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 5. CORE EXECUTION ENGINE
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="START-US VISION",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_enterprise_theme()

    st.markdown(
        """
        <div style="display: flex; align-items: center; padding: 0.8rem 0 1.2rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1.4rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(14, 165, 233, 0.4); border-radius: 10px; width: 52px; height: 52px; display: flex; align-items: center; justify-content: center; font-size: 1.7rem;">
                    ⚡
                </div>
                <div>
                    <div style="font-size: 2.35rem; font-weight: 900; letter-spacing: -0.03em; color: #f8fafc; line-height: 1.1;">
                        START-US <span style="color: #0ea5e9;">VISION</span>
                    </div>
                    <div style="font-size: 0.84rem; font-weight: 700; letter-spacing: 0.08em; color: #94a3b8; text-transform: uppercase; margin-top: 3px;">
                        Enterprise Surveillance Intelligence &amp; Re-ID Engine
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Warm up YOLO Engine in RAM (OpenVINO with PyTorch fallback)
    model = load_yolo_model("yolov8n_openvino_model", DEFAULT_MODEL_WEIGHTS)

    # 2. Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Engine Telemetry & Input")
        uploaded_file = st.file_uploader(
            "Upload Surveillance Video",
            type=SUPPORTED_VIDEO_TYPES,
            help="Supported container formats: .mp4, .mov, .avi",
        )

        st.markdown("---")
        st.markdown("#### 🎯 Active Tracking Pipeline")
        st.code(
            f"Model: {DEFAULT_MODEL_WEIGHTS}\n"
            f"Tracker: {TRACKER_CONFIG}\n"
            f"Target Class: 0 (Person)\n"
            f"Engine: YOLOv8 PyTorch MPS/CPU",
            language="yaml",
        )

        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.15,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help="Confidence threshold for human detection before feeding into tracker",
        )

    if uploaded_file is None:
        st.info("👈 Please upload a video stream via the sidebar to initialize the telemetry dashboard.")
        # Reset session state on file removal
        st.session_state.processed_session = None
        return

    # Invalidate session cache if a different file is uploaded
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("active_file_key") != file_key:
        st.session_state.active_file_key = file_key
        st.session_state.processed_session = None

    # Buffer video stream to temporary file
    temp_video_path, metadata = buffer_uploaded_video(uploaded_file)

    # -------------------------------------------------------------------------
    # TOP METRIC BAR (DYNAMIC STREAMLIT TELEMETRY GRID)
    # -------------------------------------------------------------------------
    col_m1, col_m2, col_m3 = st.columns(3)
    metric_unique_ph = col_m1.empty()
    metric_active_ph = col_m2.empty()
    metric_fps_ph = col_m3.empty()

    # Restore or set initial metrics
    if st.session_state.get("processed_session"):
        session = st.session_state.processed_session
        render_metric_card(metric_unique_ph, "TOTAL UNIQUE PEOPLE", str(session["total_unique"]), "Verified Tracking Pool", CYAN_LIGHT_HEX)
        render_metric_card(metric_active_ph, "ACTIVE TARGETS IN FRAME", "0", "Tracking Session Concluded", "#94a3b8")
        render_metric_card(metric_fps_ph, "ENGINE SPEED (FPS)", f"{session['avg_fps']:.1f}", f"Avg Processing Speed: {session['avg_fps']:.1f} FPS", EMERALD_ACCENT_HEX)
    else:
        render_metric_card(metric_unique_ph, "TOTAL UNIQUE PEOPLE", "0", "Awaiting Engine Activation", CYAN_LIGHT_HEX)
        render_metric_card(metric_active_ph, "ACTIVE TARGETS IN FRAME", "0", "0 Active Tracks", "#94a3b8")
        render_metric_card(metric_fps_ph, "ENGINE SPEED (FPS)", "0.0", f"Native Input: {metadata.fps:.1f} FPS", EMERALD_ACCENT_HEX)

    # -------------------------------------------------------------------------
    # WORKSPACE LAYOUT: DUAL COLUMN CONTAINER
    # -------------------------------------------------------------------------
    col_raw, col_proc = st.columns(2)

    with col_raw:
        raw_badge_ph = st.empty()
        raw_media_ph = st.empty()

    with col_proc:
        proc_badge_ph = st.empty()
        proc_media_ph = st.empty()

    # Control Button & Status
    st.markdown("---")
    start_col, _ = st.columns([1, 2])
    with start_col:
        run_engine = st.button("🚀 Run Real-Time Tracking Engine", type="primary", use_container_width=True)

    # High-Visibility One-Click Download Container
    download_container = st.empty()

    # Dynamic Tracking Event Table Expander
    event_expander = st.expander("📊 Target Transition Telemetry Log", expanded=True)
    with event_expander:
        event_table_ph = st.empty()

    # If already processed in this session, render the Dual Playback Matrix immediately
    if st.session_state.get("processed_session") and not run_engine:
        session = st.session_state.processed_session
        raw_badge_ph.markdown('<div class="stream-badge badge-raw">📹 Source Video Playback (Raw)</div>', unsafe_allow_html=True)
        raw_media_ph.video(session["raw_video_path"])

        proc_badge_ph.markdown('<div class="stream-badge badge-playback">⚡ AI Tracking Playback (H.264)</div>', unsafe_allow_html=True)
        proc_media_ph.video(session["web_video_path"])

        if os.path.exists(session["web_video_path"]):
            with open(session["web_video_path"], "rb") as dl_file:
                video_bytes = dl_file.read()
            base_name = os.path.splitext(uploaded_file.name)[0]
            download_container.download_button(
                label=f"📥 DOWNLOAD FULL ANALYZED TRACKING VIDEO ({len(video_bytes) / (1024*1024):.1f} MB)",
                data=video_bytes,
                file_name=f"tracked_{base_name}.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        if session["events_log"]:
            df_events_stored = pd.DataFrame(reversed(session["events_log"]))
            event_table_ph.dataframe(df_events_stored, use_container_width=True, hide_index=True)
        else:
            event_table_ph.info("No tracking events were triggered during this session.")
        return

    # If engine not running yet, show initial preview
    if not run_engine:
        raw_badge_ph.markdown('<div class="stream-badge badge-raw">📹 Input Video Stream (Source Preview)</div>', unsafe_allow_html=True)
        raw_media_ph.video(temp_video_path)

        proc_badge_ph.markdown('<div class="stream-badge badge-live">⚡ Real-Time Tracking (HUD Active)</div>', unsafe_allow_html=True)
        proc_media_ph.markdown(
            """
            <div style="height: 340px; display: flex; align-items: center; justify-content: center; 
                        background: #111827; border: 1px dashed #334155; border-radius: 8px; color: #64748b;">
                Click "Run Real-Time Tracking Engine" below to trigger live detection
            </div>
            """,
            unsafe_allow_html=True,
        )
        event_table_ph.info("No tracking events generated yet. Activate the engine to start capturing target transitions.")
        return

    # =========================================================================
    # PHASE 1: SYNCHRONIZED WORKSPACE STREAMING STATES (DURING PROCESSING)
    # =========================================================================
    raw_badge_ph.markdown('<div class="stream-badge badge-raw">🔴 Live Raw Decoder Stream</div>', unsafe_allow_html=True)
    proc_badge_ph.markdown('<div class="stream-badge badge-live">🟢 Live AI Tracking Overlay (HUD)</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Initializing OpenCV Hardware Decoder & Video Writer...")

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Failed to open buffered video stream for processing.")
        return

    pid = os.getpid()
    raw_output_path = os.path.join(tempfile.gettempdir(), f"raw_output_{pid}.mp4")
    web_output_path = os.path.join(tempfile.gettempdir(), f"web_output_{pid}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(raw_output_path, fourcc, metadata.fps, (metadata.width, metadata.height))

    unique_person_ids: Set[int] = set()
    previous_frame_ids: Set[int] = set()
    events_log: List[dict] = []
    display_id_map: dict = {}

    def get_display_id(raw_id: int) -> int:
        """Maps raw tracker IDs 1-to-1 to clean sequential display IDs starting from 1."""
        if raw_id not in display_id_map:
            display_id_map[raw_id] = len(display_id_map) + 1
        return display_id_map[raw_id]

    total_frames = max(metadata.total_frames, 1)
    frame_idx = 0
    t_start = time.time()
    rolling_fps = metadata.fps

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t_frame_start = time.time()
            timestamp_sec = frame_idx / metadata.fps

            # -----------------------------------------------------------------
            # FAST INFERENCE: classes=0 strictly filters human targets at inference
            # -----------------------------------------------------------------
            results = model.track(
                source=frame,
                tracker="fast_botsort.yaml",
                persist=True,
                classes=0,
                conf=confidence_threshold,
                verbose=False,
            )

            annotated_frame = frame.copy()
            current_frame_ids: Set[int] = set()

            # Parse Track Results
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and boxes.id is not None:
                    xyxy_coords = boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = boxes.id.int().cpu().tolist()
                    confs = boxes.conf.cpu().numpy()

                    for box, track_id, conf in zip(xyxy_coords, track_ids, confs):
                        disp_id = get_display_id(track_id)
                        x1, y1, x2, y2 = box
                        current_frame_ids.add(disp_id)
                        unique_person_ids.add(disp_id)

                        # Draw Crisp Cyber Cyan Bounding Box & HUD Corner Reticles
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), CYAN_ACCENT_BGR, 2)

                        # Corner Reticle Brackets (Defense HUD style, thick 4px)
                        corner_len = max(10, min(22, (x2 - x1) // 4, (y2 - y1) // 4))
                        # Top-Left
                        cv2.line(annotated_frame, (x1, y1), (x1 + corner_len, y1), CYAN_ACCENT_BGR, 4)
                        cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_len), CYAN_ACCENT_BGR, 4)
                        # Top-Right
                        cv2.line(annotated_frame, (x2, y1), (x2 - corner_len, y1), CYAN_ACCENT_BGR, 4)
                        cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_len), CYAN_ACCENT_BGR, 4)
                        # Bottom-Left
                        cv2.line(annotated_frame, (x1, y2), (x1 + corner_len, y2), CYAN_ACCENT_BGR, 4)
                        cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_len), CYAN_ACCENT_BGR, 4)
                        # Bottom-Right
                        cv2.line(annotated_frame, (x2, y2), (x2 - corner_len, y2), CYAN_ACCENT_BGR, 4)
                        cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_len), CYAN_ACCENT_BGR, 4)

                        # Thick, High-Visibility Capsule Badge Overlay: [ID: #01 | 94%]
                        badge_text = f"ID #{disp_id:02d} | {conf * 100:.0f}%" if disp_id < 100 else f"ID #{disp_id} | {conf * 100:.0f}%"
                        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

                        # Snug position above bounding box without clipping top of frame
                        badge_y1 = max(0, y1 - th - 10)
                        badge_y2 = y1
                        badge_x1 = x1
                        badge_x2 = min(frame.shape[1], x1 + tw + 12)

                        cv2.rectangle(annotated_frame, (badge_x1, badge_y1), (badge_x2, badge_y2), DARK_HUD_BG_BGR, -1)
                        cv2.rectangle(annotated_frame, (badge_x1, badge_y1), (badge_x2, badge_y2), CYAN_ACCENT_BGR, 2)
                        cv2.putText(
                            annotated_frame,
                            badge_text,
                            (badge_x1 + 6, badge_y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            TEXT_WHITE_BGR,
                            2,
                            cv2.LINE_AA,
                        )

            # -----------------------------------------------------------------
            # BOUNDARY TRANSITION LOGGING (ENTER / EXIT EVENTS)
            # -----------------------------------------------------------------
            entered_ids = current_frame_ids - previous_frame_ids
            exited_ids = previous_frame_ids - current_frame_ids

            for tid in sorted(entered_ids):
                events_log.append({
                    "Timestamp": f"{timestamp_sec:.2f}s",
                    "Frame": frame_idx,
                    "Target ID": f"Target #{tid:02d}" if tid < 100 else f"Target #{tid}",
                    "Event Status": "● IN-BOUND",
                    "Active Targets": len(current_frame_ids),
                })

            for tid in sorted(exited_ids):
                events_log.append({
                    "Timestamp": f"{timestamp_sec:.2f}s",
                    "Frame": frame_idx,
                    "Target ID": f"Target #{tid:02d}" if tid < 100 else f"Target #{tid}",
                    "Event Status": "○ OUT-BOUND",
                    "Active Targets": len(current_frame_ids),
                })

            previous_frame_ids = current_frame_ids

            # Dump rendered frame into output target buffer
            out.write(annotated_frame)

            # Calculate Rolling Processing Engine FPS
            frame_duration = time.time() - t_frame_start
            instant_fps = 1.0 / frame_duration if frame_duration > 0 else metadata.fps
            rolling_fps = (0.85 * rolling_fps) + (0.15 * instant_fps)

            # -----------------------------------------------------------------
            # NATIVE DESKTOP WINDOW RENDERING (WITH THREAD-SAFE FALLBACK)
            # -----------------------------------------------------------------
            try:
                cv2.imshow("START-US VISION - Real-Time AI Tracking (Press 'q' to Exit)", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except (cv2.error, Exception):
                # macOS Cocoa prevents GUI window creation from non-main worker threads (Streamlit).
                # Fallback to high-performance, throttled dual-stream browser rendering.
                if frame_idx % 2 == 0 or frame_idx == total_frames:
                    if metadata.width > 800:
                        disp_w = 720
                        disp_h = int(metadata.height * (disp_w / metadata.width))
                        disp_raw = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                        disp_proc = cv2.resize(annotated_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        disp_raw = frame
                        disp_proc = annotated_frame

                    raw_rgb = cv2.cvtColor(disp_raw, cv2.COLOR_BGR2RGB)
                    annotated_rgb = cv2.cvtColor(disp_proc, cv2.COLOR_BGR2RGB)
                    raw_media_ph.image(raw_rgb, channels="RGB", use_container_width=True)
                    proc_media_ph.image(annotated_rgb, channels="RGB", use_container_width=True)

            # Update Metric Bar & Progress Bar every 3 frames for UI responsiveness
            if frame_idx % 3 == 0 or frame_idx == total_frames:
                render_metric_card(
                    metric_unique_ph,
                    "TOTAL UNIQUE PEOPLE",
                    str(len(unique_person_ids)),
                    f"Session Pool: {len(unique_person_ids)} targets",
                    CYAN_LIGHT_HEX,
                )
                render_metric_card(
                    metric_active_ph,
                    "ACTIVE TARGETS IN FRAME",
                    str(len(current_frame_ids)),
                    f"Frame {frame_idx}/{total_frames}",
                    "#f8fafc",
                )
                render_metric_card(
                    metric_fps_ph,
                    "ENGINE SPEED (FPS)",
                    f"{rolling_fps:.1f}",
                    f"Latency: {(1000.0 / rolling_fps):.1f} ms/frame",
                    "#f59e0b" if rolling_fps < 20 else EMERALD_ACCENT_HEX,
                )

                pct = min(1.0, frame_idx / total_frames)
                progress_bar.progress(pct, text=f"Tracking Processing: Frame {frame_idx}/{total_frames} ({int(pct*100)}%)")

            # Update event table as transitions occur
            if (entered_ids or exited_ids) and events_log:
                df_events = pd.DataFrame(reversed(events_log[-50:]))
                event_table_ph.dataframe(df_events, use_container_width=True, hide_index=True)

    finally:
        # Securely release hardware pointers and file locks
        cap.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    progress_bar.progress(1.0, text="Tracking loop completed. Stabilizing and transcoding web-safe H.264 stream via FFmpeg...")

    total_elapsed = max(time.time() - t_start, 0.001)
    overall_fps = frame_idx / total_elapsed

    # Final Telemetry Metric Bar Update
    render_metric_card(metric_unique_ph, "TOTAL UNIQUE PEOPLE", str(len(unique_person_ids)), "Final Verified Pool", CYAN_LIGHT_HEX)
    render_metric_card(metric_active_ph, "ACTIVE TARGETS IN FRAME", "0", "Tracking Session Concluded", "#94a3b8")
    render_metric_card(metric_fps_ph, "ENGINE SPEED (FPS)", f"{overall_fps:.1f}", f"Avg Processing Speed: {overall_fps:.1f} FPS", EMERALD_ACCENT_HEX)

    # -------------------------------------------------------------------------
    # STABILIZATION WORKAROUND: Transmute raw OpenCV MP4V to Web-Safe H.264 MP4
    # -------------------------------------------------------------------------
    with st.spinner("🔄 Transmuting raw stream into high-performance web-safe H.264 container via FFmpeg..."):
        transcode_success = convert_to_web_safe_h264(raw_output_path, web_output_path)

    final_delivery_path = web_output_path if transcode_success else raw_output_path

    # =========================================================================
    # PHASE 2: DUAL PLAYBACK MATRIX (AFTER PROCESSING)
    # =========================================================================
    # Clear streaming frame wrappers and mount native interactive browser players
    raw_media_ph.empty()
    proc_media_ph.empty()

    raw_badge_ph.markdown('<div class="stream-badge badge-raw">📹 Source Video Playback (Raw Interactive)</div>', unsafe_allow_html=True)
    raw_media_ph.video(temp_video_path)

    proc_badge_ph.markdown('<div class="stream-badge badge-playback">⚡ AI Tracking Playback (H.264)</div>', unsafe_allow_html=True)
    proc_media_ph.video(final_delivery_path)

    if transcode_success:
        st.success(f"✅ Video processing complete! Dual playback ready at {overall_fps:.1f} FPS average.")
    else:
        st.warning("⚠️ Transmuted using raw stream fallback. Dual playback ready.")

    # =========================================================================
    # PHASE 3: RETAIN PERFORMANCE LOGS & ONE-CLICK DOWNLOAD
    # =========================================================================
    if os.path.exists(final_delivery_path):
        with open(final_delivery_path, "rb") as dl_file:
            video_bytes = dl_file.read()

        base_name = os.path.splitext(uploaded_file.name)[0]
        download_container.download_button(
            label=f"📥 DOWNLOAD FULL ANALYZED TRACKING VIDEO ({len(video_bytes) / (1024*1024):.1f} MB)",
            data=video_bytes,
            file_name=f"tracked_{base_name}.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    if events_log:
        df_events_final = pd.DataFrame(reversed(events_log))
        event_table_ph.dataframe(df_events_final, use_container_width=True, hide_index=True)
    else:
        event_table_ph.info("No boundary entry or exit events were detected during tracking.")

    # Persist session state for seamless re-render upon download button interactions
    st.session_state.processed_session = {
        "raw_video_path": temp_video_path,
        "web_video_path": final_delivery_path,
        "total_unique": len(unique_person_ids),
        "avg_fps": overall_fps,
        "events_log": events_log,
    }

    # Cleanup temporary raw intermediate file
    if os.path.exists(raw_output_path):
        try:
            os.remove(raw_output_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
