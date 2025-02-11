# app.py
from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize YOLO models
detection_model = YOLO('yolov8n.pt')
segmentation_model = YOLO('yolov8n-seg.pt')

# Global variables for webcam control
camera = None
process_frames = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi','webp'}

def process_image(file_path, model_type='detection'):
    img = cv2.imread(file_path)
    if model_type == 'detection':
        results = detection_model(img)
    else:
        results = segmentation_model(img)
    return results[0].plot()

def generate_frames(model_type='detection'):
    global camera, process_frames
    
    while process_frames:
        success, frame = camera.read()
        if not success:
            break
            
        if model_type == 'detection':
            results = detection_model(frame)
        else:
            results = segmentation_model(frame)
            
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
    global camera, process_frames
    process_frames = False
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'success'})

@app.route('/video_feed/<model_type>')
def video_feed(model_type):
    return Response(generate_frames(model_type),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'detection')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Process image
            processed_image = process_image(filepath, model_type)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
            cv2.imwrite(output_path, processed_image)
            return jsonify({
                'status': 'success',
                'type': 'image',
                'filename': f'processed_{filename}'
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