import cv2
import threading
import time
import os
import uuid
from flask import Flask, render_template, request, jsonify, Response
from models import classify_weather, detect_birds

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Session management
sessions = {}
session_lock = threading.Lock()

def cleanup_sessions():
    while True:
        with session_lock:
            current_time = time.time()
            expired = [sid for sid, session in sessions.items() 
                      if current_time - session['last_activity'] > 3600]
            for sid in expired:
                if sessions[sid]['cap'] is not None:
                    sessions[sid]['cap'].release()
                del sessions[sid]
        time.sleep(300)

threading.Thread(target=cleanup_sessions, daemon=True).start()

@app.route('/')
def index():
    return render_template('LiveModel.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'video' not in request.files:
        return jsonify({"error": "No video file"}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        
        session_id = str(uuid.uuid4())
        with session_lock:
            sessions[session_id] = {
                "weather": "Detecting...",
                "detections": [],
                "processing_active": True,
                "video_path": video_path,
                "video_size": {"width": 0, "height": 0},
                "last_activity": time.time(),
                "cap": None,
                "paused": False,
                "last_weather_update": 0
            }
            
        threading.Thread(target=process_video, args=(session_id,)).start()
        
        return jsonify({
            "message": "Processing started",
            "session_id": session_id,
            "stream_url": f"/video_feed/{session_id}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_video(session_id):
    with session_lock:
        session = sessions.get(session_id)
        if not session:
            return
        cap = cv2.VideoCapture(session['video_path'])
        session['cap'] = cap
        sessions[session_id] = session
    
    try:
        while True:
            with session_lock:
                session = sessions.get(session_id)
                if not session or not session['processing_active']:
                    break
                
                if session['paused']:
                    time.sleep(0.1)
                    continue

            ret, frame = cap.read()
            if not ret:
                break

            # Update detections
            detections = detect_birds(frame)
            
            # Update weather every 15 minutes (900 seconds)
            current_time = time.time()
            if current_time - session['last_weather_update'] > 900:
                weather = classify_weather(frame)
                with session_lock:
                    sessions[session_id]['weather'] = weather
                    sessions[session_id]['last_weather_update'] = current_time

            with session_lock:
                sessions[session_id]['detections'] = detections
                sessions[session_id]['last_activity'] = current_time
                sessions[session_id]['video_size'] = {
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                }

            time.sleep(0.03)  # ~30fps processing

    finally:
        with session_lock:
            if session_id in sessions:
                sessions[session_id]['processing_active'] = False
                sessions[session_id]['cap'].release()

@app.route('/video_feed/<session_id>')
def video_feed(session_id):
    def generate(session_id):
        with session_lock:
            session = sessions.get(session_id)
            if not session:
                return
            cap = cv2.VideoCapture(session['video_path'])
            
        while True:
            with session_lock:
                if not session['processing_active']:
                    break
            
            ret, frame = cap.read()
            if not ret:
                break
                
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        cap.release()
    
    return Response(generate(session_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<session_id>', methods=['POST'])
def control_processing(session_id):
    action = request.json.get('action')
    with session_lock:
        if session_id in sessions:
            if action == 'pause':
                sessions[session_id]['paused'] = True
            elif action == 'resume':
                sessions[session_id]['paused'] = False
            return jsonify({"status": action})
    return jsonify({"error": "Invalid session"}), 404

@app.route('/status/<session_id>')
def get_status(session_id):
    with session_lock:
        session = sessions.get(session_id)
        if session:
            return jsonify({
                "weather": session['weather'],
                "detections": session['detections'],
                "video_size": session['video_size'],
                "paused": session['paused']
            })
    return jsonify({"error": "Session not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, threaded=True)