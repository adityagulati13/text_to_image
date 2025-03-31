import streamlit as st
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("Text-to-Image Generator")
st.write(
    "Generate images from text using a pretrained Stable Diffusion model."
)

@st.cache_resource
def load_models():
    # Stable Diffusion Model
    diffusion_model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        diffusion_model.to("cuda")

    # CLIP Model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #Loads the CLIP model and its processor (used to turn text and image into embeddings)
    if torch.cuda.is_available():
        clip_model.to("cuda")

    return diffusion_model, clip_model, clip_processor   # all the models loaded by load_model() returned for future usde
diffusion_pipe, clip_model, clip_processor = load_models()

# Prompt Input
prompt = st.text_input("Enter your image prompt:")

# defining the function to analyse and score outputs
#defining a function to analyse how well the prompt and generated image is similar
def calculate_clip_score(image: Image.Image, text: str):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        #getting the embedding from model without gradient
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds.cpu().numpy()
        text_embeds = outputs.text_embeds.cpu().numpy()

    # Cosine similarity
    similarity = cosine_similarity(image_embeds, text_embeds)[0][0]  # finding the cosine simm bewtween embedding of input i.e to the clip and oof img
    semantic_accuracy = float(similarity) * 100
    return round(semantic_accuracy, 2)

# steps to get image quality--> conversion to grey scale, conversion of pixel intensities to histogram and the to probabilities and
# get pixel intensity distribution, get entropy, and use it for quality analysis
def estimate_image_quality(image: Image.Image):
    # Convert to grayscale
    gray_image = image.convert("L")
    histogram = gray_image.histogram()
    histogram_length = sum(histogram)

    samples_probability = [float(h) / histogram_length for h in histogram]
    entropy = -sum([p * np.log2(p + 1e-10) for p in samples_probability if p != 0])  # entropy

    # Normalize entropy to 0–100 scale
    quality_score = min(entropy / 8.0, 1.0) * 100
    return round(quality_score, 2)

# --- Image Generation ---

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Generate image from prompt
        result = diffusion_pipe(prompt)
        image = result.images[0]

        # Scoring
        semantic_score = calculate_clip_score(image, prompt)
        quality_score = estimate_image_quality(image)
        overall_score = round((semantic_score * 0.7 + quality_score * 0.3), 2)

        # Show Results
        st.image(image, caption=f"Generated for: '{prompt}'", use_column_width=True)
        st.metric("🔍 Semantic Accuracy (CLIP)", f"{semantic_score}%")
        st.metric("Generation Quality Score", f"{quality_score}%")
        st.success(f"Overall Confidence Score: {overall_score}%")

        # Download Button
        img_path = "generated_image.png"
        image.save(img_path)
        with open(img_path, "rb") as file:
            st.download_button(
                label="Download Image",
                data=file,
                file_name="generated_image.png",
                mime="image/png"
            )
