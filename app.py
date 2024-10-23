import gradio as gr
from diffusers import DiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-load the models but do not load them yet
models = {
    "ArtifyAI v1.1": "ImageInception/ArtifyAI-v1.1",
    "stable-diffusion-finetuned":"ImageInception/stable-diffusion-finetuned",
    "ArtifyAI v1.0": "ImageInception/ArtifyAI-v1.0"
}

# Function to load the selected model
def load_model(model_name):
    return DiffusionPipeline.from_pretrained(models[model_name], torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

# Initially load the first model
pipe = load_model("ArtifyAI v1.1")

def generate_image(prompt, selected_model):
    global pipe
    
    # Load the selected model if it's not already loaded
    if selected_model in models:
        pipe = load_model(selected_model)
    
    if device == "cuda":
        with torch.cuda.amp.autocast():
            image = pipe(prompt).images[0]
    else:
        image = pipe(prompt).images[0]
    return image

# Gradio interface
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Dropdown(choices=list(models.keys()), label="Select Model", value="ArtifyAI v1.1")
    ],
    outputs=gr.Image(type="pil"),
    title="ArtifyAI",
    description="Generate images using the selected model."
)

if __name__ == "__main__":
    demo.launch()