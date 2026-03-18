import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

st.title("🚦 AI Traffic Monitoring System")

st.write("Vehicle detection, classification and counting")

model = YOLO("yolov8n.pt")

option = st.sidebar.selectbox(
    "Choose Video Source",
    ("Use Sample Traffic Video", "Upload Traffic Video")
)

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    car_count = 0
    truck_count = 0
    bus_count = 0

    stframe = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        for r in results:

            boxes = r.boxes

            for box in boxes:

                cls = int(box.cls[0])
                label = model.names[cls]

                x1,y1,x2,y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                if label == "car":
                    car_count += 1
                elif label == "truck":
                    truck_count += 1
                elif label == "bus":
                    bus_count += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame)

    cap.release()

    return car_count, truck_count, bus_count


# SAMPLE VIDEO
if option == "Use Sample Traffic Video":

    video_path = "sample_video/traffic.mp4"

    if st.button("Start Traffic Monitoring"):

        cars, trucks, buses = process_video(video_path)

        st.subheader("Traffic Statistics")

        st.metric("Cars", cars)
        st.metric("Trucks", trucks)
        st.metric("Buses", buses)


# UPLOAD VIDEO
else:

    uploaded_video = st.file_uploader("Upload Traffic Video")

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        if st.button("Start Monitoring"):

            cars, trucks, buses = process_video(tfile.name)

            st.subheader("Traffic Statistics")

            st.metric("Cars", cars)
            st.metric("Trucks", trucks)
            st.metric("Buses", buses)
