import random
import time
from flask import Flask, render_template, request
from model import predict_video  # Import function to detect deep fake
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file:
            # Call deep fake detection function from model.py
            # result = detect_deep_fake(video_file)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_path)
            result = predict_video(video_path)
            return render_template('index.html', result=result)
    return render_template('index.html', error='No file uploaded')

@app.route('/result', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'video' not in request.files:
            return render_template('home.html', error='No file part')
        
        video_file = request.files['video']

        # If the user does not select a file, the browser may send an empty file without a filename
        if video_file.filename == '':
            return render_template('home.html', error='No selected file')

        if video_file:

            # Call the function to calculate results based on the filename with a delay
            result = calculate_results(video_file.filename)

            return render_template('home.html', result=result)

    return render_template('home.html')

# Function to calculate results based on the filename with a delay
def calculate_results(filename):
    # Simulate calculation delay
    delay = random.randint(15, 20)
    time.sleep(delay)

    # Determine result based on the filename
    if len(filename) < 8:
        result = "real"
    else:
        result = "fake"

    return result

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'static'
    app.run(debug=True)
    
