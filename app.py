from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import os
import threading
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load models
bird_model = YOLO('models/best.pt')
weather_model = load_model('models/fixed_weather_model.keras')
size_model = load_model('models/final_lstm_bird_size_model.h5')

# Configuration
WEATHER_CLASSES = ['Cloudy', 'Rainy', 'Shiny', 'Sunrise', 'Sunny']
WEATHER_INPUT_SIZE = (224, 224)
TARGET_SIZE = (640, 480)
UPDATE_INTERVAL = 30  # Update interval for secondary predictions

# Shared variables
current_weather = "Initializing..."
bird_count = 0
predicted_size = "Calculating..."
lock = threading.Lock()
video_source = None

def prepare_date_features():
    now = datetime.now()
    return {
        'day_of_year': now.timetuple().tm_yday,
        'week_of_year': now.isocalendar()[1],
        'month': now.month,
        'hour': now.hour,
        'season': (now.month % 12 + 3) // 3
    }

def predict_bird_size():
    global predicted_size
    try:
        date_features = prepare_date_features()
        # Normalize features according to your model's training
        features = np.array([
            date_features['day_of_year'] / 365,
            date_features['week_of_year'] / 52,
            date_features['month'] / 12,
            date_features['hour'] / 24,
            date_features['season'] / 4
        ]).reshape(1, 1, -1)
        
        prediction = size_model.predict(features)
        with lock:
            predicted_size = f"{prediction[0][0]:.1f} cm"
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
            'bird_size': predicted_size
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
            
            weather_thread.start()
            size_thread.start()
            
            weather_thread.join()
            size_thread.join()

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
        cv2.putText(annotated_frame, f"Avg Size: {predicted_size}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

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