import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image

# Import preprocessing and model structure
from preprocess import preprocess_image_for_inference
from train import SimpleCNN

# CIFAR-10 class labels
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image_path, model_path):
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on device: {device}")
    
    # 2. Verify model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model weights file '{model_path}' not found.")
        print("Please train the model first by running 'python train.py'.")
        return
        
    # 3. Load model and load weights
    print(f"Loading model weights from '{model_path}'...")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Load and preprocess image
    if not os.path.exists(image_path):
        print(f"Error: Input image file '{image_path}' not found.")
        return
        
    try:
        image_tensor = preprocess_image_for_inference(image_path).to(device)
    except Exception as e:
        print(f"Error reading or preprocessing image: {e}")
        return
        
    # 5. Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    # 6. Display results
    print("\n" + "="*40)
    print(f" Prediction Results for: {os.path.basename(image_path)}")
    print("="*40)
    
    # Sort predictions by confidence
    scores, indices = torch.sort(probabilities, descending=True)
    
    for i in range(len(CLASSES)):
        class_name = CLASSES[indices[i].item()]
        prob = scores[i].item()
        # Create a visual bar chart using ASCII hashes
        bar_length = int(prob * 20)
        bar = '#' * bar_length + ' ' * (20 - bar_length)
        print(f"{class_name:<12} | [{bar}] {prob*100:6.2f}%")
        
    print("="*40)
    print(f"Top Prediction: **{CLASSES[indices[0].item()].upper()}** with {scores[0].item()*100:.2f}% confidence.\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference using the trained CNN image classifier")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image to classify")
    parser.add_argument('--model_path', type=str, default='./best_cnn_model.pth', help="Path to the model weights (.pth) file")
    
    args = parser.parse_args()
    predict(args.image_path, args.model_path)
