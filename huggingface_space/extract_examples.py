"""
Script to extract example images from the dataset for the Gradio app.
Run this script after training the model to populate the examples folder.
"""

from datasets import load_dataset
import os
from PIL import Image

# Create examples directory
os.makedirs("examples", exist_ok=True)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("dpdl-benchmark/colorectal_histology")

# Label names
label_names = ['Tumor', 'Stroma', 'Complex', 'Lympho', 'Debris', 'Mucosa', 'Adipose', 'Empty']

print("Extracting example images...")

# Extract one example image per class
for label_idx, label_name in enumerate(label_names):
    # Find first image with this label
    for idx, sample in enumerate(dataset['train']):
        if sample['label'] == label_idx:
            # Save the image
            image = sample['image']
            filename = f"examples/{label_idx}_{label_name.lower()}_example.png"
            image.save(filename)
            print(f"Saved: {filename}")
            break

print("\nExample images saved successfully!")
print("You can now copy the 'examples' folder to your Hugging Face Space.")
