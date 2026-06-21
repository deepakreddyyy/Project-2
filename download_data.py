import os
from PIL import Image
from datasets import load_dataset

def download_cifar10():
    print("Checking and downloading CIFAR-10 dataset via Hugging Face...")
    
    # Download training and test sets
    train_set = load_dataset("uoft-cs/cifar10", split="train")
    test_set = load_dataset("uoft-cs/cifar10", split="test")
    
    print("CIFAR-10 loaded successfully!")
    print(f"Number of training images: {len(train_set)}")
    print(f"Number of testing images: {len(test_set)}")
    
    # Define classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create static samples directory for the web app
    samples_dir = './static/samples'
    os.makedirs(samples_dir, exist_ok=True)
    print(f"Extracting sample images for web interface to '{samples_dir}'...")
    
    # Find one sample image for each class and save it as a PNG
    saved_classes = set()
    for item in train_set:
        img = item['img']
        label = item['label']
        class_name = classes[label]
        if class_name not in saved_classes:
            # Construct filename: e.g. static/samples/dog.png
            sample_path = os.path.join(samples_dir, f"{class_name}.png")
            # Save the image
            img.save(sample_path)
            print(f"  Saved sample for '{class_name}' to {sample_path}")
            saved_classes.add(class_name)
            
        if len(saved_classes) == len(classes):
            break
            
    print("Data download and sample extraction complete!")

if __name__ == '__main__':
    download_cifar10()
