import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('mask_detection_model.h5')

# Define labels
labels = {0: 'No Mask', 1: 'Mask'}

# Access the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera; change to 1 or 2 for external cameras

while True:
    ret, frame = cap.read()  # Capture each frame from the camera
    if not ret:
        print("Failed to grab frame. Exiting...")
        break
    
    # Preprocess the frame
    resized_frame = cv2.resize(frame, (128, 128))  # Resize frame to model input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to 0-1
    input_data = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

    # Predict using the trained model
    prediction = model.predict(input_data)
    label = labels[np.argmax(prediction)]  # Get label with highest probability

    # Display prediction on the frame
    color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (20, 20), (200, 80), color, 2)

    # Show the frame with prediction
    cv2.imshow('Face Mask Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

