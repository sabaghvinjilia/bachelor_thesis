from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

is_running = True

def detect_and_predict_mask(frame, maskNet):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # OpenCV's Haar Cascade classifier [Face Detection]
    # https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    locs = []
    face_list = []

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = preprocess_input(img_to_array(face))

        face_list.append(face)
        locs.append((x, y, x + w, y + h))

    preds = maskNet.predict(np.array(face_list, dtype="float32"), batch_size=32) if face_list else []
    return locs, preds

maskNet = load_model("mask.h5")

# cv2.VideoCapture(0) - Phone Camera
# cv2.VideoCapture(1) - Builtin Camera
cap = cv2.VideoCapture(1)

while is_running:
    ret, frame = cap.read()
    if not ret:
        is_running = False

    frame = cv2.resize(frame, (800, 600))
    locs, preds = detect_and_predict_mask(frame, maskNet)

    for (box, pred) in zip(locs, preds):
        startX, startY, endX, endY = box
        mask, withoutMask = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = f"{label}: {max(mask, withoutMask) * 100:.0f}%"

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    print('Press q to quit.')

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        is_running = False

cap.release()
cv2.destroyAllWindows()