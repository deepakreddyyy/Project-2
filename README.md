# NeuroClass: Custom CNN Image Classifier

NeuroClass is a premium, modern Convolutional Neural Network (CNN) Image Classifier trained on the CIFAR-10 dataset. It features a custom 6-layer PyTorch neural network, a Flask backend API, and a beautiful glassmorphic web user interface.

---

## 🚀 Key Features

*   **Custom CNN Architecture**: A 6-layer PyTorch CNN (with BatchNorm, MaxPool, Dropout, and Dense layers) achieving **69.25% validation accuracy**.
*   **Premium Web Interface**: A modern glassmorphic UI featuring live file upload, real-time predictions, probability distribution charts, and a pixelation preview showcasing "what the CNN sees" (32x32 spatial downsampling).
*   **Multi-Set Test Gallery**: Includes 5 pre-extracted sample image sets (50 total images, 5 per class) which are classified correctly by our trained model with high confidence.
*   **Command Line Tool**: A flexible CLI script to classify images and print ASCII-based prediction confidence charts directly in the terminal.
*   **Complete Lifecycle Scripts**: Standard modular scripts for downloading, preprocessing, training, and predicting.

---

## 📁 File Structure

*   `app.py` - Flask backend server.
*   `train.py` - Custom CNN definition, training loop, and evaluation.
*   `preprocess.py` - Image transforms, augmentations, and dataset loading wrappers.
*   `predict.py` - Command-line prediction tool.
*   `download_data.py` - Dataset retrieval via Hugging Face.
*   `requirements.txt` - Project dependencies.
*   `best_cnn_model.pth` - Pre-trained model weights.
*   `static/` - Glassmorphism UI (HTML, CSS, JS, and pre-extracted samples).
*   `cnn_image_classification.ipynb` - Step-by-step Jupyter Notebook.
*   `report.md` - Technical project report.

---

## 🛠️ Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Web Server
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

### 3. Run Command-Line Inference
```bash
python predict.py --image_path static/samples/airplane_1.png
```

### 4. Train/Retrain the Model
```bash
python train.py
```
