from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import tempfile
from collections import defaultdict
import json
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize YOLO models
detection_model = YOLO('yolov8n.pt')
segmentation_model = YOLO('yolov8n-seg.pt')

# Global variables
camera = None
process_frames = False
confidence_threshold = 0.5
last_frame_analytics = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'webp'}

def get_analytics_from_results(results):
    """Extract analytics data from YOLO results"""
    analytics = {
        'total_objects': 0,
        'average_confidence': 0,
        'classes': [],
        'class_counts': [],
        'detections': []
    }
    
    if not results or not results[0].boxes:
        return analytics
    
    # Get boxes and confidence scores
    boxes = results[0].boxes
    class_names = results[0].names
    
    # Count objects by class
    class_counts = defaultdict(int)
    total_confidence = 0
    
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = class_names[class_id]
        
        class_counts[class_name] += 1
        total_confidence += confidence
        
        analytics['detections'].append({
            'class': class_name,
            'confidence': confidence,
            'bbox': box.xyxy[0].tolist()
        })
    
    analytics['total_objects'] = len(boxes)
    analytics['average_confidence'] = total_confidence / len(boxes) if len(boxes) > 0 else 0
    analytics['classes'] = list(class_counts.keys())
    analytics['class_counts'] = list(class_counts.values())
    
    return analytics

def process_image(file_path, model_type='detection', conf=0.5):
    """Process image with analytics"""
    img = cv2.imread(file_path)
    
    if model_type == 'detection':
        results = detection_model(img, conf=conf)
    else:
        results = segmentation_model(img, conf=conf)
    
    # Get analytics before plotting
    analytics = get_analytics_from_results(results)
    
    # Plot the results
    processed_img = results[0].plot()
    
    return processed_img, analytics

def generate_frames(model_type='detection'):
    global camera, process_frames, confidence_threshold, last_frame_analytics
    
    while process_frames:
        success, frame = camera.read()
        if not success:
            break
        
        if model_type == 'detection':
            results = detection_model(frame, conf=confidence_threshold)
        else:
            results = segmentation_model(frame, conf=confidence_threshold)
        
        # Update analytics for the current frame
        last_frame_analytics = get_analytics_from_results(results)
        
        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_webcam')
def start_webcam():
    global camera, process_frames
    if camera is None:
        camera = cv2.VideoCapture(0)
        process_frames = True
    return jsonify({'status': 'success'})

@app.route('/stop_webcam')
def stop_webcam():
    global camera, process_frames, last_frame_analytics
    process_frames = False
    if camera is not None:
        camera.release()
        camera = None
    last_frame_analytics = None
    return jsonify({'status': 'success'})

@app.route('/video_feed/<model_type>')
def video_feed(model_type):
    global confidence_threshold
    confidence_threshold = float(request.args.get('confidence', 0.5)) / 100
    return Response(generate_frames(model_type),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_frame_analytics')
def get_frame_analytics():
    """Endpoint to get analytics for the current frame"""
    global last_frame_analytics
    return jsonify(last_frame_analytics or {})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'detection')
    confidence = float(request.form.get('confidence_threshold', 50)) / 100
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            # Process image with analytics
            processed_image, analytics = process_image(filepath, model_type, confidence)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
            cv2.imwrite(output_path, processed_image)
            
            return jsonify({
                'status': 'success',
                'type': 'image',
                'filename': f'processed_{filename}',
                'analytics': analytics
            })
        else:
            # For video files
            return jsonify({
                'status': 'success',
                'type': 'video',
                'filename': filename
            })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)