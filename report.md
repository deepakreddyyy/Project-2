# Project Report: Image Classification using Convolutional Neural Networks (CNNs)

## 1. Project Overview
This project focuses on building a complete end-to-end Computer Vision image classifier utilizing Convolutional Neural Networks (CNNs). The model is trained on the standard **CIFAR-10** dataset to classify 32x32 color images into one of 10 distinct classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

The primary goals of the project were:
- Design a high-performance deep learning CNN architecture in **PyTorch**.
- Implement standard computer vision preprocessing pipelines including normalization and data augmentation.
- Understand and mitigate overfitting using regularization techniques (Dropout, Batch Normalization).
- Evaluate the model using loss/accuracy curves and a multi-class confusion matrix.
- Deploy the model as an interactive web-based classification tool and command-line application.

---

## 2. Technology Stack
- **Programming Language**: Python 3.13
- **Deep Learning Framework**: PyTorch (with CUDA support for NVIDIA GPU training)
- **Data Visualization**: Matplotlib, Seaborn
- **Image Processing**: Pillow (PIL)
- **Web App Server**: Flask, Flask-CORS
- **Frontend Stack**: HTML5, Vanilla CSS3 (Custom Glassmorphism Design), Vanilla Javascript

---

## 3. Dataset Exploration & Preprocessing
The **CIFAR-10** dataset contains 60,000 color images of size 32x32 pixels, divided into 50,000 training images and 10,000 testing images.

### Preprocessing Pipelines
Two distinct preprocessing pipelines were created:
1. **Training Transform**:
   - **Random Crop** with padding of 4: Generates translation variations of the object to prevent spatial overfitting.
   - **Random Horizontal Flip**: Flips images with a 50% probability to teach the model mirror-invariance (e.g., a dog facing left is still a dog when facing right).
   - **Random Rotation (15°)**: Introduces rotational invariance.
   - **ToTensor**: Converts the PIL image pixels (range [0, 255]) to floating-point tensors (range [0.0, 1.0]).
   - **Normalization**: Standardizes color channels by subtracting the mean and dividing by the standard deviation computed across the CIFAR-10 dataset:
     $$\text{Mean} = [0.4914, 0.4822, 0.4465], \quad \text{Std} = [0.2470, 0.2435, 0.2616]$$
     This shifts inputs to a mean of zero and standard deviation of one, accelerating optimization convergence.

2. **Inference Transform**:
   - **Resize (32x32)**: Ensures uploaded user images match the required input size.
   - **ToTensor**: Normal pixel scaling.
   - **Normalization**: Uses the exact same channel-wise CIFAR-10 stats.

---

## 4. CNN Architecture Design
To achieve reliable multi-class accuracy, a custom VGG-style deep architecture was designed with 6 convolutional layers and 2 dense layers:

```
[Input: 3x32x32 Image]
        │
   [Conv Block 1]  ───► Conv2d(3 -> 32) ──► BatchNorm2d ──► Conv2d(32 -> 32) ──► BatchNorm2d ──► MaxPool2d(2x2) ──► Dropout(0.25)
        │
   [Conv Block 2]  ───► Conv2d(32 -> 64) ──► BatchNorm2d ──► Conv2d(64 -> 64) ──► BatchNorm2d ──► MaxPool2d(2x2) ──► Dropout(0.25)
        │
   [Conv Block 3]  ───► Conv2d(64 -> 128) ──► BatchNorm2d ──► Conv2d(128 -> 128) ──► BatchNorm2d ──► MaxPool2d(2x2) ──► Dropout(0.40)
        │
    [Flatten]      ───► Reshape to 1D vector (128 * 4 * 4 = 2048 elements)
        │
  [Dense Block 1]  ───► Linear(2048 -> 512) ──► BatchNorm1d ──► ReLU ──► Dropout(0.50)
        │
   [Output Dense]  ───► Linear(512 -> 10 classes) ──► Raw Logits
```

### Key Architectural Concepts
1. **Convolutional Layers (`Conv2d`)**: Extract local spatial features. The kernel size is fixed at $3\times3$ with padding=1, which allows stacking multiple layers to build larger receptive fields while maintaining fewer parameters.
2. **Batch Normalization (`BatchNorm2d`, `BatchNorm1d`)**: Normalizes the activations of the previous layer at each batch, reducing internal covariate shift. This enables higher learning rates, acts as a mild regularizer, and stabilizes training.
3. **Max Pooling (`MaxPool2d`)**: Performs spatial downsampling by taking the maximum value in a $2\times2$ grid. This reduces the number of operations and parameters, making the representation translation-invariant.
4. **Dropout (`Dropout`)**: A powerful regularization technique that randomly sets a fraction of activations (25%, 40%, or 50% in this model) to zero during training. This prevents units from co-adapting too much, forcing the network to learn redundant and robust representations.
5. **Fully Connected Layers (`Linear`)**: Combines high-level features extracted by conv layers to output a distribution across the 10 classes.

---

## 5. Training Strategy
- **Loss Function**: `CrossEntropyLoss` (automatically applies LogSoftmax internally).
- **Optimizer**: `Adam` (adaptive learning rate optimizer) with a base learning rate of $0.001$.
- **Batch Size**: 64.
- **Split**: 80% Training (40,000 images), 20% Validation (10,000 images).
- **Device Management**: Automatically detects if a GPU is available. If training is run on a CPU, the script automatically reduces training set size to 10,000 images and 5 epochs to prevent system lockups while achieving representative learning. On GPU (NVIDIA RTX 3050), it trains on the full 50,000 dataset for 15 epochs.
- **Model Checkpointing**: The script tracks validation accuracy and only saves the weights when validation performance improves.

---

## 6. Evaluation and Results

*Note: Visual curves and confusion matrix are saved in the `static/` directory.*

### Insights from Confusion Matrix
- **High Performance Categories**: Vehicles (e.g., `automobile`, `truck`, `ship`) typically achieve high accuracy (above 85%) because of distinct borders, solid structures, and backgrounds (e.g. water for ships).
- **Common Confusions**: The model frequently displays minor confusion between semantically similar biological classes:
  - `dog` vs. `cat` (similar visual shapes and features).
  - `automobile` vs. `truck` (similar metallic structural boxes).
  - `bird` vs. `airplane` (objects against sky backgrounds).

---

## 7. Interactive Interfaces

### Command-line Prediction (`predict.py`)
Provides rapid terminal-based testing:
```bash
python predict.py --image_path static/samples/dog.png
```
This prints a clean ASCII bar chart displaying the confidence scores of all classes and lists the top classification.

### Flask Web Application (`app.py`)
Features a polished dark mode interface utilizing glassmorphic layouts:
1. **Drag-and-Drop Image Uploader**: Allows dragging local JPG/PNG images.
2. **Sample Image Gallery**: Preloaded with CIFAR-10 images for immediate testing with a single click.
3. **Inference Stats**: Shows image resolution and processing hardware (e.g., CUDA GPU vs CPU).
4. **"What the CNN Sees" Canvas**: Downsamples and displays the input image as a 32x32 pixelated representation, educating users on model inputs.
5. **Dynamic Predictions Chart**: Renders animated horizontal progress bars colored by confidence.

---

## 8. Conclusion and Future Improvements
This project successfully demonstrates the efficacy of deep convolutional layers in processing spatial image grids. By applying batch normalization and dropout, we successfully mitigated overfitting, which is common in deep networks.

**Future Enhancements**:
- **Learning Rate Scheduling**: Implementing a learning rate scheduler (e.g., CosineAnnealingLR) to refine convergence in later training epochs.
- **Advanced Augmentation**: Introducing CutMix or RandAugment to force more generalized boundary learning.
- **Feature Map Visualizations**: Creating a layer extraction endpoint to show activation heatmaps (Grad-CAM) directly on the web interface.
