from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load YOLOv8 model
model = YOLO('models/best.pt')

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Video path storage
video_path = None

# Route to upload video
@app.route('/', methods=['GET', 'POST'])
def upload_video():
    global video_path

    if request.method == 'POST':
        if 'video' not in request.files:
            return "No file uploaded!"
        file = request.files['video']
        if file.filename == '':
            return "No selected file!"
        if file:
            # Save the video to uploads folder
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)
            return redirect(url_for('index'))

    return render_template('upload.html')

# Route to display the video
@app.route('/index')
def index():
    if not video_path:
        return redirect(url_for('upload_video'))
    return render_template('index.html')

# Video streaming with object detection
def generate_frames():
    cap = cv2.VideoCapture(video_path)
    target_width, target_height = 640, 480

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # Perform object detection
        results = model(resized_frame)

        # Draw bounding boxes and labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                # Draw rectangle and label
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(resized_frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', resized_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
