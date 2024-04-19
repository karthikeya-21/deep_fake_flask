from flask import Flask, render_template, request
from model import predict_video  # Import function to detect deep fake
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'static'
    app.run(debug=True)
    
