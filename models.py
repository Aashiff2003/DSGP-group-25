import cv2
import numpy as np
from keras.models import load_model
from ultralytics import YOLO

weather_model = load_model('model/fixed_weather_model.keras')
bird_model = YOLO('C:/Users/aashi/Downloads/Geethmi/Backend/best.pt')

def classify_weather(frame):
    try:
        resized = cv2.resize(frame, (224, 224))
        normalized = resized / 255.0
        pred = weather_model.predict(np.expand_dims(normalized, axis=0))
        classes = ['Cloudy', 'Rainy', 'Sunny']
        return classes[np.argmax(pred)]
    except Exception as e:
        print(f"Weather error: {e}")
        return "Unknown"

def detect_birds(frame):
    try:
        results = bird_model(frame)
        detections = []
        h, w = frame.shape[:2]
        
        for result in results:
            for box in result.boxes:
                if box.conf.item() > 0.3:
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    detections.append({
                        "x": int(round(x1 * w)),
                        "y": int(round(y1 * h)),
                        "w": int(round((x2 - x1) * w)),
                        "h": int(round((y2 - y1) * h)),
                        "confidence": float(box.conf.item())
                    })
        return detections
    except Exception as e:
        print(f"Detection error: {e}")
        return []