from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the YOLO model
model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Run YOLO inference
    results = model(image)

    # Process results
    detections = []
    bird_count = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = float(box.conf[0])  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            # Assuming '0' is the bird class in your dataset
            if class_id == 0:
                bird_count += 1
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(image, f'Bird {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detections.append({
                "class_id": class_id,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

    # Save the output image
    output_path = "output.jpg"
    cv2.imwrite(output_path, image)

    return jsonify({
        "bird_count": bird_count,
        "detections": detections,
        "image_url": "/output"
    })

@app.route('/output', methods=['GET'])
def get_output_image():
    return send_file("output.jpg", mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
