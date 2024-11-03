import os
import sys
import webbrowser
from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from threading import Timer

# Tắt warnings không cần thiết
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MODEL_PATH'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'animal_classifier_model.h5')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Đảm bảo thư mục uploads tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Định nghĩa các class động vật
class_names = ["Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", 
               "Elephant", "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"]

# Load model
try:
    model = tf.keras.models.load_model(app.config['MODEL_PATH'])
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def open_browser():
    webbrowser.open('http://127.0.0.1:5000/')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400
        
    if file and allowed_file(file.filename):
        try:
            # Lưu file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Xử lý ảnh
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            # Dự đoán
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Xóa file sau khi xử lý
            os.remove(filepath)
            
            if confidence > 0.5:
                result = {
                    'animal': class_names[predicted_class],
                    'confidence': round(confidence * 100, 2),
                    'status': 'success'
                }
            else:
                result = {
                    'animal': 'Unknown',
                    'confidence': 0,
                    'status': 'low_confidence'
                }
                
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500
    
    return jsonify({'error': 'File không hợp lệ', 'status': 'invalid_file'}), 400

if __name__ == '__main__':
    Timer(1.5, open_browser).start()
    app.run(debug=True, use_reloader=False)