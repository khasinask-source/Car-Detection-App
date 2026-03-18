import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from tracker import Tracker

st.set_page_config(page_title="Smart Traffic Monitoring", layout="wide")
st.title("🚦 Smart Traffic Monitoring System")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

with st.spinner("Loading AI model..."):
    model = load_model()

option = st.sidebar.selectbox(
    "Choose Video Source",
    ("Use Sample Traffic Video", "Upload Traffic Video")
)

# ── Tuning sliders in sidebar ──────────────────────────────────────────
frame_skip   = st.sidebar.slider("Process every Nth frame", 1, 6, 2)
infer_width  = st.sidebar.slider("Inference width (px)", 320, 1280, 640, step=64)
conf_thresh  = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4)

def run_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    tracker = Tracker()
    vehicle_ids = set()
    line_y = 400
    stframe = st.empty()
    count_placeholder = st.empty()

    frame_count = 0
    last_boxes_ids = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ── Fix 1: Skip frames ─────────────────────────────────────────
        if frame_count % frame_skip != 0:
            continue

        # ── Fix 2: Resize for inference only ──────────────────────────
        h, w = frame.shape[:2]
        scale = infer_width / w
        small = cv2.resize(frame, (infer_width, int(h * scale)))

        # ── Fix 3: conf filter + no verbose spam ──────────────────────
        results = model(small, conf=conf_thresh, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                cls   = int(box.cls[0])
                label = model.names[cls]
                if label in ["car", "truck", "bus"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # scale coords back to original frame size
                    x1 = int(x1 / scale); y1 = int(y1 / scale)
                    x2 = int(x2 / scale); y2 = int(y2 / scale)
                    detections.append([x1, y1, x2 - x1, y2 - y1])

        last_boxes_ids = tracker.update(detections)

        for box in last_boxes_ids:
            x, y, bw, bh, id = box
            cx = x + bw // 2
            cy = y + bh // 2
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, str(id), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if cy > line_y:
                vehicle_ids.add(id)

        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)

        # ── Fix 4: Display at reduced size ────────────────────────────
        display = cv2.resize(frame, (960, int(h * 960 / w)))
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        stframe.image(display, use_container_width=True)
        count_placeholder.metric("🚗 Vehicles Counted", len(vehicle_ids))

    cap.release()
    st.success(f"✅ Done! Total vehicles counted: {len(vehicle_ids)}")

if option == "Use Sample Traffic Video":
    video_path = "sample_video/traffic.mp4"
    if st.button("Start Traffic Monitoring"):
        run_detection(video_path)
else:
    uploaded_video = st.file_uploader("Upload Traffic Video")
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        if st.button("Start Monitoring"):
            run_detection(tfile.name)
