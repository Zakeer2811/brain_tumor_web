
import os
import cv2
import gdown
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
from ultralytics import YOLO

# ─── New: Google Drive Download Logic ────────────────────────────────────────
# 1) RAW Drive file ID:
DRIVE_FILE_ID = '1B0FfStSYKtdQ8Hh9UfyXWMHLzvbcid41'
# 2) Construct the “export=download” URL for gdown:
DRIVE_URL = f'https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}'
# 3) Local filename we want:
LOCAL_MODEL_PATH = 'best.onnx'

# If the ONNX model isn't already on disk, download it from Drive:
if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"→ Downloading ONNX model from Google Drive to '{LOCAL_MODEL_PATH}' …")
    # gdown will handle large-file tokens automatically.
    gdown.download(DRIVE_URL, LOCAL_MODEL_PATH, quiet=False)
    print("✔ Download complete.")

# ─── End of Download Logic ────────────────────────────────────────────────────

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = LOCAL_MODEL_PATH    # now points to the downloaded file

# Create the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load the YOLO ONNX model using Ultralytics
model = YOLO(MODEL_PATH)


# Enhanced HTML templates with modern styling
index_html = '''
<!doctype html>
<html lang="en">
<head>
    <title>Brain Tumor MRI Detection</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 500px;
            width: 90%;
            transform: translateY(0);
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .container:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 50px rgba(0,0,0,0.15);
        }
        
        h1 {
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: 700;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: titleGlow 3s ease-in-out infinite alternate;
        }
        
        @keyframes titleGlow {
            from { filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.3)); }
            to { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 0.5)); }
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px 20px;
            margin: 30px 0;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .upload-area::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(102, 126, 234, 0.1), transparent);
            transform: rotate(45deg);
            transition: all 0.6s ease;
            opacity: 0;
        }
        
        .upload-area:hover::before {
            animation: shimmer 1.5s ease-in-out infinite;
            opacity: 1;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: rgba(102, 126, 234, 0.05);
            transform: scale(1.02);
        }
        
        .upload-icon {
            font-size: 3em;
            color: #667eea;
            margin-bottom: 15px;
            display: block;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover .upload-icon {
            transform: scale(1.1);
            color: #764ba2;
        }
        
        input[type="file"] {
            opacity: 0;
            position: absolute;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }
        
        .upload-text {
            color: #666;
            font-size: 1.1em;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }
        
        .upload-subtext {
            color: #999;
            font-size: 0.9em;
            position: relative;
            z-index: 1;
        }
        
        .submit-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 25px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            margin-top: 20px;
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .submit-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .submit-btn:hover::before {
            left: 100%;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        }
        
        .submit-btn:active {
            transform: translateY(0);
        }
        
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(102, 126, 234, 0.2);
        }
        
        .footer p {
            color: #666;
            font-size: 0.9em;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .medical-badge {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .loading {
            display: none;
            margin-top: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 600px) {
            .container {
                margin: 20px;
                padding: 30px 20px;
            }
            
            h1 {
                font-size: 2em;
            }
            
            .upload-area {
                padding: 30px 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1> Brain Tumor MRI Detection</h1>
        <form method="post" enctype="multipart/form-data" id="uploadForm">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">🧠</div>
                <div class="upload-text">Upload MRI Scan for Detection</div>
                <div class="upload-subtext">Supports T1, T2, T1CE, FLAIR sequences • JPG, PNG, DICOM</div>
                <input type="file" name="file" accept="image/*" id="fileInput" onchange="handleFileSelect(this)">
            </div>
            <div id="fileName" style="color: #667eea; margin: 10px 0; font-weight: 600;"></div>
            <button type="submit" class="submit-btn" id="submitBtn">
                <span id="btnText">🔬 Analyze MRI Scan</span>
            </button>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div style="color: #667eea; font-weight: 600;">Processing your image...</div>
            </div>
        </form>
        <div class="footer">
            <p>Medical Imaging Analysis • <span class="medical-badge">Research Project</span></p>
        </div>
    </div>
    
    <script>
        function handleFileSelect(input) {
            if (input.files && input.files[0]) {
                const fileName = input.files[0].name;
                document.getElementById('fileName').textContent = `Selected: ${fileName}`;
                document.getElementById('btnText').innerHTML = '🎯 Detect Tumor';
            }
        }
        
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files || !fileInput.files[0]) {
                e.preventDefault();
                alert('Please select an image first!');
                return;
            }
            
            // Show loading animation
            document.getElementById('loading').style.display = 'block';
            document.getElementById('submitBtn').style.display = 'none';
        });
        
        // Drag and drop functionality
        const uploadArea = document.querySelector('.upload-area');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight(e) {
            uploadArea.style.borderColor = '#764ba2';
            uploadArea.style.background = 'rgba(102, 126, 234, 0.1)';
        }
        
        function unhighlight(e) {
            uploadArea.style.borderColor = '#667eea';
            uploadArea.style.background = 'transparent';
        }
        
        uploadArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                document.getElementById('fileInput').files = files;
                handleFileSelect(document.getElementById('fileInput'));
            }
        }
    </script>
</body>
</html>
'''

result_html = '''
<!doctype html>
<html lang="en">
<head>
    <title>Detection Results | Brain Tumor MRI</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            animation: slideIn 0.8s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: 700;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }
        
        .success-badge {
            background: rgba(255, 255, 255, 0.9);
            color: #28a745;
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 0.6em;
            font-weight: 600;
            border: 2px solid #28a745;
            animation: pulse 2s infinite;
            box-shadow: 0 2px 10px rgba(40, 167, 69, 0.2);
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .image-container {
            text-align: center;
            margin: 40px 0;
            position: relative;
        }
        
        .result-image {
            max-width: 100%;
            height: auto;
            border-radius: 15px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 3px solid transparent;
            background: linear-gradient(white, white) padding-box,
                        linear-gradient(45deg, #667eea, #764ba2) border-box;
        }
        
        .result-image:hover {
            transform: scale(1.02);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        
        .image-overlay {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.95);
            color: #e74c3c;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            backdrop-filter: blur(10px);
            border: 2px solid #e74c3c;
            animation: fadeInScale 1s ease-out 0.5s both;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.2);
        }
        
        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: scale(0.8);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        .actions {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 40px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            position: relative;
            overflow: hidden;
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .btn:hover::before {
            left: 100%;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            box-shadow: 0 10px 20px rgba(40, 167, 69, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(40, 167, 69, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            border: 1px solid rgba(102, 126, 234, 0.2);
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.1);
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        
        @media (max-width: 600px) {
            .container {
                margin: 10px;
                padding: 20px;
            }
            
            h1 {
                font-size: 2em;
                flex-direction: column;
                gap: 10px;
            }
            
            .actions {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 100%;
                max-width: 300px;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>
            Detection Results
            
        </h1>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">🧠</div>
                <div class="stat-label">MRI Analysis</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">📊</div>
                <div class="stat-label">Detailed Results</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">⚡</div>
                <div class="stat-label">Quick Processing</div>
            </div>
        </div>
        
        <div class="image-container">
            <img src="{{ url_for('static', filename='results/' + filename) }}" 
                 alt="Brain Tumor Detection Results" 
                 class="result-image"
                 onload="this.style.animation='fadeInScale 1s ease-out'">
            <div class="image-overlay">🔬 Analyzed</div>
        </div>
        
        <div class="actions">
            <a href="{{ url_for('index') }}" class="btn btn-primary">
                🔄 Analyze Another Scan
            </a>
            <a href="{{ url_for('static', filename='results/' + filename) }}" 
               download class="btn btn-secondary">
                💾 Download Result
            </a>
        </div>
    </div>
    
    <script>
        // Add some interactive animations
        document.addEventListener('DOMContentLoaded', function() {
            // Animate stat cards
            const statCards = document.querySelectorAll('.stat-card');
            statCards.forEach((card, index) => {
                card.style.animation = `slideIn 0.6s ease-out ${index * 0.1}s both`;
            });
            
            // Add click effect to buttons
            const buttons = document.querySelectorAll('.btn');
            buttons.forEach(button => {
                button.addEventListener('click', function(e) {
                    const ripple = document.createElement('div');
                    ripple.style.cssText = `
                        position: absolute;
                        border-radius: 50%;
                        background: rgba(255,255,255,0.6);
                        width: 100px;
                        height: 100px;
                        left: ${e.offsetX - 50}px;
                        top: ${e.offsetY - 50}px;
                        animation: ripple 0.6s ease-out;
                        pointer-events: none;
                    `;
                    this.appendChild(ripple);
                    setTimeout(() => ripple.remove(), 600);
                });
            });
        });
        
        // Add ripple effect keyframes
        const style = document.createElement('style');
        style.textContent = `
            @keyframes ripple {
                from {
                    transform: scale(0);
                    opacity: 1;
                }
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check for file in request
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file
            filename = file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Run detection on the uploaded image
            results = model.predict(source=upload_path, imgsz=640)
            
            # Force manual saving of the result image:
            # Get the plotted result as a numpy array
            img_result = results[0].plot()
            # Define the output path (static/results folder with same filename)
            output_path = os.path.join(app.config['RESULT_FOLDER'], filename)
            cv2.imwrite(output_path, img_result)

            # Redirect to the result page
            return redirect(url_for('result', filename=filename))
    return render_template_string(index_html)

@app.route('/result/<filename>')
def result(filename):
    return render_template_string(result_html, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def create_folders():
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
create_folders()
if __name__ == '__main__':
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

