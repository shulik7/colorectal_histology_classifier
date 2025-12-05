import gradio as gr
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os
import spaces

# Model configuration
MODEL_NAME = "shulik7/vit-colorectal-histology"  # Replace with your model path on Hugging Face

# Class labels
LABELS = {
    0: "Tumor",
    1: "Stroma", 
    2: "Complex",
    3: "Lympho",
    4: "Debris",
    5: "Mucosa",
    6: "Adipose",
    7: "Empty"
}

# Label descriptions
LABEL_DESCRIPTIONS = {
    "Tumor": "Cancerous tissue",
    "Stroma": "Connective tissue",
    "Complex": "Complex/mixed tissue patterns",
    "Lympho": "Lymphocytes (immune cells)",
    "Debris": "Cellular debris",
    "Mucosa": "Mucosal tissue",
    "Adipose": "Fat tissue",
    "Empty": "Empty/background regions"
}

# Global variables to store model and processor
processor = None
model = None

def load_model():
    """
    Load the model and processor.
    This function is called once when initializing the app.
    """
    global processor, model
    
    print("Loading model and processor...")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()  # Set to evaluation mode
    print("Model and processor loaded successfully!")
    
    return processor, model


@spaces.GPU(duration=10)  # Allocate GPU for 10 seconds per inference
def predict(image):
    """
    Make prediction on the input image using Zero GPU.
    The @spaces.GPU decorator ensures the model is loaded to GPU only during inference.
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        dict: Prediction probabilities for each class
        str: Formatted prediction text
    """
    global processor, model
    
    # Load model if not already loaded
    if processor is None or model is None:
        processor, model = load_model()
    
    # Ensure image is PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Move results back to CPU and convert to dict
    probabilities = probabilities.cpu().numpy()
    results = {LABELS[i]: float(probabilities[i]) for i in range(len(LABELS))}
    
    # Get top prediction
    top_class_idx = probabilities.argmax()
    top_class = LABELS[top_class_idx]
    top_prob = probabilities[top_class_idx]
    
    # Format prediction text
    prediction_text = f"""
    ## 🔬 Prediction Results
    
    **Predicted Tissue Type:** **{top_class}**
    
    **Description:** {LABEL_DESCRIPTIONS[top_class]}
    
    **Confidence:** {top_prob:.2%}
    
    ---
    
    ### All Class Probabilities:
    """
    
    for label, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
        prediction_text += f"\n- **{label}**: {prob:.2%}"
    
    # Offload model from GPU to free memory for other users
    model.to("cpu")
    torch.cuda.empty_cache()
    
    return results, prediction_text


# Initialize model at startup (on CPU to save GPU resources)
print("Initializing application...")
processor, model = load_model()
print("Application ready!")

# Create example images list
example_dir = "examples"
examples = []
if os.path.exists(example_dir):
    example_files = sorted([os.path.join(example_dir, f) for f in os.listdir(example_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))])
    examples = [[f] for f in example_files]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 Colorectal Histology Tissue Classifier
        
        Upload a histology image to classify it into one of 8 tissue types using a fine-tuned Vision Transformer model.
        
        **Tissue Types:**
        - 🔴 **Tumor**: Cancerous tissue
        - 🟢 **Stroma**: Connective tissue
        - 🟣 **Complex**: Complex/mixed tissue patterns
        - 🔵 **Lympho**: Lymphocytes (immune cells)
        - 🟤 **Debris**: Cellular debris
        - 🟡 **Mucosa**: Mucosal tissue
        - 🟠 **Adipose**: Fat tissue
        - ⚪ **Empty**: Empty/background regions
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Histology Image",
                type="pil",
                height=400
            )
            
            predict_btn = gr.Button("🔍 Classify Tissue", variant="primary", size="lg")
            
            gr.Markdown("### 📝 Example Images")
            gr.Markdown("Click on an example below to try the classifier:")
            
        with gr.Column(scale=1):
            output_plot = gr.Label(
                label="Prediction Probabilities",
                num_top_classes=8
            )
            
            output_text = gr.Markdown(label="Detailed Results")
    
    # Add examples if available
    if examples:
        gr.Examples(
            examples=examples,
            inputs=input_image,
            label="Try these example images:"
        )
    else:
        gr.Markdown(
            """
            ⚠️ **No example images found.**
            
            To add example images, create an `examples/` folder and add histology images with filenames like:
            - `tumor_example.png`
            - `stroma_example.png`
            - etc.
            """
        )
    
    # Connect the button to the prediction function
    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_plot, output_text]
    )
    
    gr.Markdown(
        """
        ---
        
        ## 📊 Model Information
        
        - **Base Model**: [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
        - **Dataset**: [Colorectal Histology](https://huggingface.co/datasets/dpdl-benchmark/colorectal_histology)
        - **Input Size**: 224×224 pixels (automatically resized)
        - **Architecture**: Vision Transformer (ViT-Base)
        
        ## 💻 About This App
        
        This application uses **Hugging Face Zero GPU** to provide fast inference while minimizing resource usage.
        The model is loaded to GPU only during prediction and immediately offloaded afterward.
        
        ## 🔗 Links
        
        - [Model Card](https://huggingface.co/shulik7/vit-colorectal-histology)
        - [GitHub Repository](https://github.com/shulik7/colorectal_histology-classifier)
        - [Training Notebook](https://github.com/shulik7/colorectal_histology-classifier/blob/main/vit_finetuning.ipynb)
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.queue()  # Enable queuing for better performance
    demo.launch()
