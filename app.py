import cv2
from datetime import datetime

# Load the face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open the webcam
camera = cv2.VideoCapture(0)

# Open log file
log_file = open("face_log.txt", "a")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not access the camera")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(50, 50)
    )

    # Draw rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # Display face count
    cv2.putText(
        frame,
        "Faces Detected: " + str(len(faces)),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display status
    if len(faces) > 0:
        status = "FACE DETECTED"
        
        # Save detection in log file
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(
            current_time + " - Face Detected - "
            + str(len(faces)) + " face(s)\n"
        )
        log_file.flush()

    else:
        status = "NO FACE DETECTED"

    cv2.putText(
        frame,
        status,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cv2.putText(
        frame,
        current_time,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Show camera window
    cv2.imshow("Real-Time Face Detection", frame)

    # Keyboard input
    key = cv2.waitKey(1) & 0xFF

    # Press S to save image
    if key == ord("s"):
        cv2.imwrite("captured_face.jpg", frame)
        print("Image saved as captured_face.jpg")

    # Press Q to exit
    if key == ord("q"):
        break

# Close everything
camera.release()
log_file.close()
cv2.destroyAllWindows()