from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model  # type: ignore
import os
import threading
from datetime import datetime
import pandas as pd
from joblib import load
import time
import webbrowser
import torch
from tensorflow import config as tf_config
import pymysql
from sqlalchemy import create_engine

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# MySQL Configuration for phpMyAdmin
DB_NAME = 'FalconEye'
DB_USER = 'root'
DB_PASSWORD = ''
DB_HOST = 'localhost'

# Function to create database if not exists
def create_database():
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Database '{DB_NAME}' created successfully or already exists")
    except Exception as e:
        print(f"Error creating database: {str(e)}")

# Create database first
create_database()

# Configure SQLAlchemy to use the created database
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the AlertRecord model
class AlertRecord(db.Model):
    __tablename__ = 'alert_records'  # Explicit table name for MySQL compatibility
    
    id = db.Column(db.Integer, primary_key=True)
    weather = db.Column(db.String(255), nullable=False)
    bird_size = db.Column(db.String(255), nullable=False)
    bird_quantity = db.Column(db.Integer, nullable=False)
    alert_level = db.Column(db.String(255), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# GPU Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Configure TensorFlow to use GPU
gpus = tf_config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf_config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load models with GPU support
bird_model = YOLO('models/best.pt').to(device)
weather_model = load_model('models/final_weather_classification_model.keras')
size_model = load_model('models/final_lstm_bird_size_model.h5')
rf_model = load('models/random_forest_model.joblib')

# Configuration
WEATHER_CLASSES = ['alien_test', 'cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
WEATHER_INPUT_SIZE = (224, 224)
TARGET_SIZE = (640, 480)
UPDATE_INTERVAL = 30  # Update interval for secondary predictions

# Hardcoded scaling parameters
FEATURE_RANGES = {
    'Migration_start_year': (2020, 2030),
    'Migration_start_month': (1, 12),
    'Migration_end_month': (1, 12),
    'GPS_xx': (0, 100),
    'GPS_yy': (0, 100),
    'Feature_6': (0, 1),
    'Feature_7': (0, 1),
}

# Label mappings
LABEL_MAPPING = {
    0: "Small",
    1: "Medium",
    2: "Large",
}

WEATHER_TO_CONDITIONS = {
    'shine': 'ConditionsSky_No Cloud',
    'sunrise': 'ConditionsSky_No Cloud',
    'cloudy': 'ConditionsSky_Overcast',
    'foggy': 'ConditionsSky_Overcast',
    'rainy': 'ConditionsSky_Overcast',
    'alien_test': 'ConditionsSky_Some Cloud',
}

RISK_MAPPING = {
    0: "Low",
    1: "Moderate",
    2: "High",
}

# Shared variables
current_weather = "N/A"
bird_count = 0
predicted_size = "N/A"
alert_level = "N/A"
lock = threading.Lock()
current_video_source = None
processing_enabled = False

# Fluctuation tracking
last_state = {
    'weather': "N/A",
    'bird_size': "N/A",
    'alert_level': "N/A"
}
current_open_record = None

def prepare_date_features():
    now = datetime.now()
    return {
        'Migration_start_year': now.year,
        'Migration_start_month': now.month,
        'Migration_end_month': (now.month % 12) + 1,
        'GPS_xx': 7.8731,
        'GPS_yy': 80.7718,
        'Feature_6': 0.5,
        'Feature_7': 0.5,
    }

def manual_scale(features):
    """Hardcoded scaling logic"""
    scaled_features = []
    for i, key in enumerate(FEATURE_RANGES):
        min_val, max_val = FEATURE_RANGES[key]
        scaled_value = (features[i] - min_val) / (max_val - min_val)
        scaled_features.append(scaled_value)
    return np.array(scaled_features)

@app.route('/')
def login():
    return render_template('Signin.html')

@app.route('/Home')
def home():
    return render_template('Home.html')

@app.route('/Report')
def report():
    return render_template('Report.html')

@app.route('/Account')
def account():
    return render_template('Account.html')

@app.route('/Insights', methods=['GET', 'POST'])
def insights():
    global current_video_source, processing_enabled
    processing_enabled = False  # Reset processing when returning to insights
    
    if request.method == 'POST':
        if 'webcam' in request.form:
            current_video_source = 'webcam'
            processing_enabled = True
            return redirect(url_for('index'))
        elif 'upload' in request.form:
            file = request.files['video']
            if file and file.filename != '':
                current_video_source = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(current_video_source)
                processing_enabled = True
                return redirect(url_for('index'))
    return render_template('Insights.html')

@app.route('/index')
def index():
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
    global bird_count, current_weather, predicted_size, alert_level
    cap = None
    
    if current_video_source == 'webcam':
        cap = cv2.VideoCapture(0)
    elif current_video_source and os.path.exists(current_video_source):
        cap = cv2.VideoCapture(current_video_source)
    
    if not cap or not cap.isOpened():
        return

    frame_counter = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            if current_video_source == 'webcam': 
                continue  # Keep trying for webcam
            else: 
                break  # Stop for uploaded videos

        if processing_enabled:
            resized_frame = cv2.resize(frame, TARGET_SIZE)
            
            # Perform object detection
            frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float().to(device)
            frame_tensor = frame_tensor.unsqueeze(0) / 255.0
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                results = bird_model(frame_tensor)
            
            current_count = len(results[0].boxes) if results else 0
            with lock:
                bird_count = current_count

            # Update secondary predictions periodically
            frame_counter += 1
            if frame_counter % UPDATE_INTERVAL == 0:
                # Capture current bird_count
                with lock:
                    current_bird_count = bird_count

                # Get predictions
                weather_pred = process_weather(frame)
                size_pred = predict_bird_size()
                alert_pred = predict_alert_level(current_bird_count, weather_pred, size_pred)

                # Update global state
                with lock:
                    if (weather_pred != current_weather or
                        size_pred != predicted_size or
                        alert_pred != alert_level):
                        
                        current_weather = weather_pred
                        predicted_size = size_pred
                        alert_level = alert_pred
                        handle_fluctuation(current_bird_count)

            # Create annotated frame
            annotated_frame = resized_frame.copy()
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw status text
            cv2.putText(annotated_frame, f"Weather: {current_weather}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Birds: {bird_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Size Category: {predicted_size}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"Alert Level: {alert_level}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            annotated_frame = cv2.resize(frame, TARGET_SIZE)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    torch.cuda.empty_cache()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_processing')
def stop_processing():
    global processing_enabled
    processing_enabled = False
    return redirect(url_for('insights'))

def process_weather(frame):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = cv2.resize(rgb_frame, WEATHER_INPUT_SIZE)
        processed = processed.astype('float32') / 255.0
        
        predictions = weather_model.predict(np.expand_dims(processed, axis=0), verbose=0)
        predicted_class = WEATHER_CLASSES[np.argmax(predictions)]
        confidence = np.max(predictions)
        
        return predicted_class if confidence > 0.6 else current_weather
    except Exception as e:
        print(f"Weather prediction error: {str(e)}")
        return current_weather

def predict_bird_size():
    try:
        date_features = prepare_date_features()
        input_data = np.array([
            date_features['Migration_start_year'],
            date_features['Migration_start_month'],
            date_features['Migration_end_month'],
            date_features['GPS_xx'],
            date_features['GPS_yy'],
            date_features['Feature_6'],
            date_features['Feature_7']
        ])
        
        scaled_features = manual_scale(input_data)
        X_input = scaled_features.reshape((1, 1, scaled_features.shape[0]))
        
        prediction = size_model.predict(X_input, verbose=0)
        return LABEL_MAPPING[np.argmax(prediction)]
    except Exception as e:
        print(f"Size prediction error: {str(e)}")
        return predicted_size

def predict_alert_level(bird_count, weather, size):
    try:
        conditions_sky = WEATHER_TO_CONDITIONS.get(weather, 'ConditionsSky_Some Cloud')
        size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
        wildlife_size = size_mapping.get(size, 1)
        
        input_data = {
            'NumberStruckActual': bird_count,
            'WildlifeSize': wildlife_size,
            'ConditionsSky_No Cloud': 1 if conditions_sky == 'ConditionsSky_No Cloud' else 0,
            'ConditionsSky_Overcast': 1 if conditions_sky == 'ConditionsSky_Overcast' else 0,
            'ConditionsSky_Some Cloud': 1 if conditions_sky == 'ConditionsSky_Some Cloud' else 0,
        }
        
        input_df = pd.DataFrame([input_data])
        prediction = rf_model.predict(input_df)
        return RISK_MAPPING.get(prediction[0], "Moderate")
    except Exception as e:
        print(f"Alert level prediction error: {str(e)}")
        return "Moderate"

def handle_fluctuation(bird_count):
    global last_state, current_open_record
    current_state = {
        'weather': current_weather,
        'bird_size': predicted_size,
        'alert_level': alert_level,
    }
    if current_state != last_state:
        now = datetime.utcnow()
        try:
            with app.app_context():
                # Close existing open record
                if current_open_record:
                    current_open_record.end_time = now
                    db.session.commit()
                # Create new record
                new_record = AlertRecord(
                    weather=current_weather,
                    bird_size=predicted_size,
                    bird_quantity=bird_count,
                    alert_level=alert_level,
                    start_time=now
                )
                db.session.add(new_record)
                db.session.commit()
                current_open_record = new_record
                last_state = current_state.copy()
        except Exception as e:
            print(f"Database error: {str(e)}")

def open_dashboard():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == '__main__':
    # Create upload folder if not exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Create database and tables
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully")
        except Exception as e:
            print(f"Error creating tables: {str(e)}")
    
    # Open browser and run app
    threading.Thread(target=open_dashboard, daemon=True).start()
    app.run(debug=True)