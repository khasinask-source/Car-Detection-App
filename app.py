import streamlit as st
import cv2
import tempfile
import os

st.set_page_config(page_title="Car Detection AI", layout="wide")

st.title("🚗 Car Detection AI App")
st.write("Detect cars in videos using OpenCV Haar Cascade")

# Load classifier
car_classifier = cv2.CascadeClassifier("haarcascades/haarcascade_car.xml")

# Sidebar
option = st.sidebar.selectbox(
    "Choose Video Source",
    ("Use Sample Video", "Upload Your Own Video")
)

# -----------------------------
# Function to process video
# -----------------------------
def process_video(input_path, output_path):

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    car_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cars = car_classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30,30)
        )

        for (x,y,w,h) in cars:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            car_count += 1

        out.write(frame)

    cap.release()
    out.release()

    return car_count


# -----------------------------
# SAMPLE VIDEO
# -----------------------------
if option == "Use Sample Video":

    video_path = "sample_video/cars.mp4"

    st.subheader("Sample Car Detection Video")

    if st.button("▶ Start Detection"):

        st.info("Processing video... Please wait.")

        output_video = "output_sample.mp4"

        car_count = process_video(video_path, output_video)

        st.success("Detection Completed!")

        st.subheader("Processed Video")
        st.video(output_video)

        st.metric("Cars Detected", car_count)

        with open(output_video, "rb") as file:
            st.download_button(
                "⬇ Download Processed Video",
                file,
                file_name="car_detection_output.mp4"
            )


# -----------------------------
# UPLOAD VIDEO
# -----------------------------
elif option == "Upload Your Own Video":

    uploaded_video = st.file_uploader(
        "Upload a video (Recommended under 1 minute)",
        type=["mp4","avi","mov"]
    )

    if uploaded_video is not None:

        if st.button("▶ Start Detection"):

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            st.info("Processing video... Please wait.")

            output_video = "output_upload.mp4"

            car_count = process_video(tfile.name, output_video)

            st.success("Detection Completed!")

            st.subheader("Processed Video")
            st.video(output_video)

            st.metric("Cars Detected", car_count)

            with open(output_video, "rb") as file:
                st.download_button(
                    "⬇ Download Processed Video",
                    file,
                    file_name="car_detection_output.mp4"
                )
