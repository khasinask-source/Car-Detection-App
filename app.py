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

def run_detection(video_path):

    cap = cv2.VideoCapture(video_path)

    tracker = Tracker()
    vehicle_ids = set()

    line_y = 400

    stframe = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        detections = []

        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])
                label = model.names[cls]

                if label in ["car","truck","bus"]:

                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    w = x2-x1
                    h = y2-y1

                    detections.append([x1,y1,w,h])

        boxes_ids = tracker.update(detections)

        for box in boxes_ids:

            x,y,w,h,id = box

            cx = x+w//2
            cy = y+h//2

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,str(id),(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

            if cy > line_y:
                vehicle_ids.add(id)

        cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(0,0,255),3)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        stframe.image(frame)

    cap.release()

# Sample video
if option == "Use Sample Traffic Video":

    video_path = "sample_video/traffic.mp4"

    if st.button("Start Traffic Monitoring"):
        run_detection(video_path)

# Upload video
else:

    uploaded_video = st.file_uploader("Upload Traffic Video")

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        if st.button("Start Monitoring"):
            run_detection(tfile.name)
