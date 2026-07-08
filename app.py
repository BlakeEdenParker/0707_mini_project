from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = os.path.join(BASE_DIR, 'rps_model.h5')
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define class labels (must match the order from flow_from_directory)
# flow_from_directory sorts subdirectories alphanumerically: paper, rock, scissors
# So index 0 = paper, index 1 = rock, index 2 = scissors
CLASSES = {0: '보 (Paper)', 1: '바위 (Rock)', 2: '가위 (Scissors)'}

def predict_image(filepath):
    # Load and resize image to 32x32
    img = load_img(filepath, target_size=(32, 32))
    # Convert image to numpy array
    img_array = img_to_array(img)
    # Scale image pixels to [0, 1] (this is our "scaler")
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    return CLASSES[predicted_class_idx], confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            prediction, confidence = predict_image(filepath)
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return jsonify({
                'prediction': prediction,
                'confidence': f"{confidence*100:.2f}%"
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
