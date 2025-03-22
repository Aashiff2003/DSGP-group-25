import cv2
import numpy as np
from keras.models import load_model

# Load models
weather_model = load_model('model/fixed_weather_model.keras')
bird_model = load_model('bird_model.h5')

# Preprocessing functions
def preprocess_frame(frame, target_size=(224, 224)):
    frame = cv2.resize(frame, target_size)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Classify Weather
def classify_weather(frame):
    try:
        preprocessed_frame = preprocess_frame(frame)
        prediction = weather_model.predict(preprocessed_frame)
        classes = ['Sunny', 'Rainy', 'Cloudy', 'Snowy']
        return classes[np.argmax(prediction)]
    except Exception as e:
        print(f"Error in weather classification: {e}")
        return "Unknown"

# Detect Birds

# Assuming that you are using a model that can give bounding box coordinates for bird detection
def detect_birds(frame):
    try:
        preprocessed_frame = preprocess_frame(frame)
        prediction = bird_model.predict(preprocessed_frame)

        # If the model provides bounding boxes (e.g., for object detection models)
        # Assuming prediction[0] contains the bounding boxes and confidence scores
        if prediction[0][0] > 0.5:  # You can adjust this threshold as needed
            # Example: assuming prediction provides (x, y, w, h) for bounding box
            x, y, w, h = prediction[0][1:5]  # Example, adjust based on your model
            # Draw a bounding box around the detected bird
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green color, thickness 2
            return "Bird Detected"
        else:
            return "No Bird Detected"
    except Exception as e:
        print(f"Error in bird detection: {e}")
        return "Detection Error"

