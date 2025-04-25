import cv2
import dlib
import time
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from inference import get_model

# Model configuration
model = get_model(model_id="eyes-classification/1", api_key="xxx")

# Paths for face detection and landmark model
landmark_model_path = "D:/AIProject/shape_predictor_68_face_landmarks.dat"

# Create a directory to save captured images if it doesn't exist
output_folder = "D:/AIProject/captured_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dlib's facial landmark predictor
predictor = dlib.shape_predictor(landmark_model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access the webcam.")
    exit()

frame_count = 0
capture_interval = 0.01  # Interval between captures in seconds
img_size = 100  # Expected input size for the eye detection model
alpha = 0.7  # Smoothing factor

# Threshold to determine drowsiness
drowsy_threshold = 5  # Number of consecutive frames predicted as "Drowsy" to mark the person as drowsy
drowsy_count = 0
prev_label = "Non-Drowsy"

def crop_eyes(img, landmarks):
    """
    Extracts the eye regions using the detected facial landmarks.
    """
    left_eye_points = landmarks[36:42]
    right_eye_points = landmarks[42:48]

    # Find bounding box around both eyes
    all_points = np.concatenate((left_eye_points, right_eye_points), axis=0)
    x, y, w, h = cv2.boundingRect(all_points)

    # Expand the bounding box slightly
    margin = 20
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)

    eyes_region = img[y1:y2, x1:x2]
    return eyes_region

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Cannot capture frame from webcam.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Get the coordinates of the first detected face
        x, y, w, h = faces[0]
        padding = 20

        # Smooth the bounding box coordinates
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        face_only = frame[y1:y2, x1:x2]

        # Convert the cropped face to grayscale for dlib processing
        gray_face = cv2.cvtColor(face_only, cv2.COLOR_BGR2GRAY)

        # Use dlib to detect face and landmarks within the cropped region
        dlib_faces = dlib.rectangle(0, 0, face_only.shape[1], face_only.shape[0])
        landmarks = predictor(gray_face, dlib_faces)
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract eye regions using the landmarks
        eyes_image = crop_eyes(face_only, landmarks_np)

        # Preprocess the eyes region for the model prediction
        if eyes_image is not None and eyes_image.size > 0:
            eyes_resized = cv2.resize(eyes_image, (img_size, img_size))
            eyes_resized = eyes_resized.astype("float32") / 255.0
            eyes_array = img_to_array(eyes_resized)
            eyes_array = np.expand_dims(eyes_array, axis=0)

            # Make a prediction using your model
            prediction = model.infer(eyes_image)[0]
            current_label = prediction.top

            # Update drowsy count
            if current_label == "Drowsy":
                drowsy_count += 1
            else:
                drowsy_count = 0

            # Check if the person is drowsy
            if drowsy_count >= drowsy_threshold and prev_label == "Drowsy":
                print("Person is considered Drowsy.")
                final_label = "Drowsy"
            else:
                final_label = "Non-Drowsy"

            # Save the eyes region with the label in the filename
            frame_path = os.path.join(output_folder, f"eyes_{frame_count}_{final_label}.jpg")
            cv2.imwrite(frame_path, eyes_image)
            print(f"Saved eyes region: {frame_path}")
            frame_count += 1

            # Update the previous label
            prev_label = current_label

    # Display the resulting frame
    cv2.imshow('Drowsiness Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Delay for the specified interval
    time.sleep(capture_interval)

# Release resources
cap.release()
cv2.destroyAllWindows()
