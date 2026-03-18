import streamlit as st
import cv2
import tempfile

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
# SAMPLE VIDEO
# -----------------------------
if option == "Use Sample Video":

    st.subheader("Sample Car Detection Video")

    if st.button("▶ Start Car Detection"):

        video_path = "sample_video/sample video.mp4"

        cap = cv2.VideoCapture(video_path)

        stframe = st.empty()

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

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            stframe.image(frame,channels="RGB")

        cap.release()


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

            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()

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

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                stframe.image(frame,channels="RGB")

            cap.release()
