import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import preprocessing and model structure
from preprocess import preprocess_image_for_inference
from train import SimpleCNN

# CIFAR-10 class labels
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Initialize Flask app
app = Flask(__name__)
# Enable CORS
CORS(app)

MODEL_PATH = "best_cnn_model.pth"
model = None
device = None

def load_model():
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading trained CNN model onto device: {device}...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file '{MODEL_PATH}' not found. Prediction API will run in mock mode until a model is trained.")
        return
        
    try:
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# Serve the static frontend files
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Serve other static files (CSS, JS, sample images) from the static folder
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Endpoint to predict uploaded images or sample gallery images
@app.route('/predict', methods=['POST'])
def predict():
    global model, device
    
    # Reload model if it was trained after startup
    if model is None:
        if os.path.exists(MODEL_PATH):
            load_model()
        else:
            return jsonify({
                "status": "error", 
                "message": "Model not trained yet. Please run 'python train.py' to train the model first."
            }), 500

    img = None
    img_name = "Uploaded Image"
    
    # Check if this is a sample image request (JSON with sample name)
    if request.is_json:
        data = request.get_json(silent=True)
        if data and 'sample_class' in data:
            sample_class = data['sample_class']
            sample_set = data.get('sample_set', 1)
            # Try to load set specific image: e.g., airplane_2.png
            sample_path = os.path.join('static', 'samples', f"{sample_class}_{sample_set}.png")
            if not os.path.exists(sample_path):
                # Fallback to default name: airplane.png
                sample_path = os.path.join('static', 'samples', f"{sample_class}.png")
                
            if os.path.exists(sample_path):
                img = Image.open(sample_path)
                img_name = f"Sample {sample_class.capitalize()} (Set {sample_set})"
            else:
                return jsonify({"status": "error", "message": f"Sample image for '{sample_class}' not found."}), 400
    
    # Otherwise, check for uploaded file
    if img is None:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image file or sample class provided."}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected."}), 400
        try:
            img = Image.open(file.stream)
            img_name = file.filename
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid image file: {str(e)}"}), 400

    try:
        # Save original dimensions
        orig_width, orig_height = img.size
        
        # Preprocess and move tensor to device
        tensor = preprocess_image_for_inference(img).to(device)
        
        # Run model prediction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
        # Format prediction results
        scores, indices = torch.sort(probabilities, descending=True)
        
        top_prediction = CLASSES[indices[0].item()]
        top_confidence = scores[0].item() * 100.0
        
        probs_dict = []
        for i in range(len(CLASSES)):
            c_name = CLASSES[indices[i].item()]
            prob_pct = scores[i].item() * 100.0
            probs_dict.append({
                "class": c_name,
                "confidence": round(prob_pct, 2)
            })
            
        return jsonify({
            "status": "success",
            "filename": img_name,
            "prediction": top_prediction,
            "confidence": round(top_confidence, 2),
            "probabilities": probs_dict,
            "dimensions": f"{orig_width}x{orig_height}",
            "device": str(device)
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    # Initial load attempt
    load_model()
    # Run the web server
    print("Starting Flask web server on port 5000...")
    app.run(host='127.0.0.1', port=5000, debug=True)
