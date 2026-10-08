import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Face Detection System",
    page_icon="👤"
)

st.title("👤 Cloud-Based Face Detection System")

st.write("Real-time face detection using OpenCV")

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read uploaded image
    file_bytes = uploaded_file.read()

    image = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(80, 80)
    )

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Display result
    st.image(
        image_rgb,
        caption="Face Detection Result",
        use_container_width=True
    )

    # Download result
    success, encoded_image = cv2.imencode(
        ".jpg",
        image
    )

    if success:
        st.download_button(
            label="📥 Download Detection Result",
            data=encoded_image.tobytes(),
            file_name="face_detection_result.jpg",
            mime="image/jpeg"
        )

    # Detection result
    st.subheader("Detection Result")

    if len(faces) > 0:
        st.success("Face Detected")
        st.write("Number of Faces:", len(faces))
    else:
        st.warning("No Face Detected")