from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import os
import threading
from datetime import datetime
import pandas as pd
from joblib import load
from db import db, connect_db
from database import (
    create_database, add_user, password_check,
    add_report, fetch_reports
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
connect_db(app)
create_database(app)

# Load models
bird_model = YOLO('models/best.pt')
weather_model = load_model('models/final_weather_classification_model.keras')
size_model = load_model('models/final_lstm_bird_size_model.h5')
rf_model = load('models/random_forest_model.joblib')  # Load Random Forest model

# Configuration
WEATHER_CLASSES = ['alien_test', 'cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
WEATHER_INPUT_SIZE = (224, 224)
TARGET_SIZE = (640, 480)
UPDATE_INTERVAL = 30  # Update interval for secondary predictions

# Hardcoded scaling parameters (replace with actual values from training)
FEATURE_RANGES = {
    'Migration_start_year': (2020, 2030),  # Example range
    'Migration_start_month': (1, 12),
    'Migration_end_month': (1, 12),
    'GPS_xx': (0, 100),  # Example range
    'GPS_yy': (0, 100),  # Example range
    'Feature_6': (0, 1),  # Placeholder range
    'Feature_7': (0, 1),  # Placeholder range
}

# Hardcoded label mapping
LABEL_MAPPING = {
    0: "Small",
    1: "Medium",
    2: "Large",
}

# Map weather predictions to ConditionsSky
WEATHER_TO_CONDITIONS = {
    'shine': 'No Cloud',
    'sunrise': 'No Cloud',
    'cloudy': 'Overcast',
    'foggy': 'Overcast',
    'rainy': 'Overcast',
    'alien_test': 'Some Cloud',
}

# Risk mapping
RISK_MAPPING = {
    0: "Low",
    1: "Moderate",
    2: "High",
}

# Shared variables
current_weather = "Initializing..."
bird_count = 0
predicted_size = "Calculating..."
alert_level = "Calculating..."
lock = threading.Lock()
video_source = None

def prepare_date_features():
    now = datetime.now()
    return {
        'Migration_start_year': now.year,
        'Migration_start_month': now.month,
        'Migration_end_month': (now.month % 12) + 1,  # Handle December wrap-around
        'GPS_xx': 7.8731,  # Default GPS coordinates
        'GPS_yy': 80.7718,
        'Feature_6': 0.5,  # Placeholder feature
        'Feature_7': 0.5,  # Placeholder feature
    }

def manual_scale(features):
    """Hardcoded scaling logic to replace scaler.pkl."""
    scaled_features = []
    for i, key in enumerate(FEATURE_RANGES):
        min_val, max_val = FEATURE_RANGES[key]
        scaled_value = (features[i] - min_val) / (max_val - min_val)
        scaled_features.append(scaled_value)
    return np.array(scaled_features)

def predict_bird_size():
    global predicted_size
    try:
        date_features = prepare_date_features()
        input_data = np.array([date_features['Migration_start_year'],
                               date_features['Migration_start_month'],
                               date_features['Migration_end_month'],
                               date_features['GPS_xx'],
                               date_features['GPS_yy'],
                               date_features['Feature_6'],
                               date_features['Feature_7']])
        
        # Normalize features using hardcoded scaling logic
        scaled_features = manual_scale(input_data)
        
        # Reshape for LSTM
        X_input = scaled_features.reshape((1, 1, scaled_features.shape[0]))
        
        # Make prediction
        prediction = size_model.predict(X_input)
        predicted_class = LABEL_MAPPING[np.argmax(prediction)]
        
        with lock:
            predicted_size = predicted_class
    except Exception as e:
        print(f"Size prediction error: {str(e)}")

def process_weather(frame):
    global current_weather
    try:
        processed = cv2.resize(frame, WEATHER_INPUT_SIZE)
        processed = processed / 255.0
        predictions = weather_model.predict(np.expand_dims(processed, axis=0))
        with lock:
            current_weather = WEATHER_CLASSES[np.argmax(predictions)]
    except Exception as e:
        print(f"Weather prediction error: {str(e)}")

def predict_alert_level():
    global alert_level
    try:
        # Map current_weather to ConditionsSky
        conditions_sky = WEATHER_TO_CONDITIONS.get(current_weather, 'Some Cloud')
        
        # Map predicted_size to WildlifeSize encoding
        size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
        wildlife_size = size_mapping.get(predicted_size, 1)  # Default to Medium
        
        # Prepare input for Random Forest model
        input_data = {
            'NumberStruckActual': bird_count,
            'WildlifeSize': wildlife_size,
            'ConditionsSky_No Cloud': 1 if conditions_sky == 'No Cloud' else 0,
            'ConditionsSky_Overcast': 1 if conditions_sky == 'Overcast' else 0,
            'ConditionsSky_Some Cloud': 1 if conditions_sky == 'Some Cloud' else 0,
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Predict alert level
        prediction = rf_model.predict(input_df)
        alert_level = RISK_MAPPING.get(prediction[0], "Moderate")  # Default to Moderate
        
    except Exception as e:
        print(f"Alert level prediction error: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    global video_source
    if request.method == 'POST':
        if 'webcam' in request.form:
            video_source = 'webcam'
            return redirect(url_for('index'))
        elif 'upload' in request.form:
            file = request.files['video']
            if file and file.filename != '':
                video_source = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(video_source)
                return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/index')
def index():
    if not video_source:
        return redirect(url_for('upload_video'))
    return render_template('index.html')

@app.route('/stats')
def get_stats():
    with lock:
        return jsonify({
            'weather': current_weather,
            'bird_count': bird_count,
            'bird_size': predicted_size,
            'alert_level': alert_level,
        })

def generate_frames():
    global bird_count
    cap = cv2.VideoCapture(0) if video_source == 'webcam' else cv2.VideoCapture(video_source)
    frame_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            if video_source == 'webcam': continue
            else: break

        # Resize frame
        resized_frame = cv2.resize(frame, TARGET_SIZE)
        
        # Bird detection
        results = bird_model(resized_frame)
        current_count = len(results[0].boxes) if results else 0
        
        # Update counts
        with lock:
            bird_count = current_count

        # Periodic updates
        frame_counter += 1
        if frame_counter % UPDATE_INTERVAL == 0:
            weather_thread = threading.Thread(target=process_weather, args=(frame,))
            size_thread = threading.Thread(target=predict_bird_size)
            alert_thread = threading.Thread(target=predict_alert_level)
            
            weather_thread.start()
            size_thread.start()
            alert_thread.start()
            
            weather_thread.join()
            size_thread.join()
            alert_thread.join()

        # Draw detections and info
        annotated_frame = resized_frame.copy()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Information overlay
        cv2.putText(annotated_frame, f"Weather: {current_weather}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Birds: {bird_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Size Category: {predicted_size}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Alert Level: {alert_level}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
