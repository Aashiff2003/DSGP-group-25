import cv2
import threading
import time
from flask import Flask, render_template, request, jsonify, Response
from models import classify_weather, detect_birds
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Shared Variables
current_weather = "Unknown"
current_bird_status = "No Detection"

# Route for video upload and display
@app.route('/')
def index():
    return render_template('LiveModel.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video part"})

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({"error": "No selected file"})

    if video_file:
        uploads_dir = app.config['UPLOAD_FOLDER']
        os.makedirs(uploads_dir, exist_ok=True)

        video_path = os.path.join(uploads_dir, video_file.filename)
        video_file.save(video_path)
        return jsonify({"video_url": f"/{video_path}"})

# Generate video frames for bird detection
def generate_frames(video_path):
    global current_bird_status
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform bird detection (this now draws a bounding box around detected birds)
        current_bird_status = detect_birds(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Video streaming route
@app.route('/video_stream/<filename>')
def video_stream(filename):
    return Response(generate_frames(os.path.join(app.config['UPLOAD_FOLDER'], filename)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Weather classification every 5 minutes
def weather_classification(video_path):
    global current_weather
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_weather = classify_weather(frame)
        print(f"Weather Update: {current_weather}")
        time.sleep(300)  # Run every 5 minutes

# Start parallel threads for weather classification and bird detection
def start_threads(video_path):
    threading.Thread(target=weather_classification, args=(video_path,), daemon=True).start()
    print("Weather classification thread started.")

# API to fetch results
@app.route('/results', methods=['GET'])
def results():
    return jsonify({"weather": current_weather, "bird_status": current_bird_status})

if __name__ == '__main__':
    app.run(debug=True)