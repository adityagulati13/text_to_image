# A streamlit app using the pretrained model (“runwayml/stable-diffusion-v1-5”) and diffusers, with metrices to analyse the output.

1.	Application Workflow
	User Prompt Input- Users begin by entering a natural language text prompt that describes the desired scene . This prompt forms the foundation for the image generation process.  
	Image Generation-The refined prompt is passed into the StableDiffusionPipeline from Hugging Face’s diffusers library, which generates a corresponding image.
	Semantic Accuracy Evaluation (CLIP)- Once the image is generated, the app evaluates how semantically aligned the image is to the original prompt using OpenAI’s CLIP (Contrastive Language–Image Pretraining) model.
This involves:
Encoding the text prompt into a vector (text_embeds).Encoding the generated image into another vector (image_embeds).Computing the cosine similarity between these embeddings.
	Image Quality Evaluation-To assess visual richness and detail, the image undergoes an entropy-based quality analysis:
The image is converted to grayscale, A pixel intensity histogram is computed
Shannon entropy is calculated from the distribution of pixel intensities
	Overall Confidence Score- An overall confidence score is calculated by combining semantic and visual metrics:
	Overall Confidence= (0.7×Semantic Accuracy)+(0.3×Image Quality Score) 
	A download button is provided to save the image in .png format for personal use, sharing, or further processing.

2.	Code Architecture
	load_models()- Initializes and caches Stable Diffusion and CLIP models using @st.cache_resource. Automatically selects GPU if available to accelerate inference.
	calculate_clip_score()- Computes cosine similarity between image and text embeddings using CLIP. Returns a semantic alignment percentage.
	estimate_image_quality()- Computes grayscale entropy of the image and scales it to estimate perceived quality. Acts as a proxy for detail and sharpness.
	st.button("Generate")- Triggers the generation pipeline: prompt refinement → image creation → scoring.



## Demo
https://youtu.be/w6iDbDGKt7Y
https://texttoimg-6mf6oulb4cpsrgcyzegdwy.streamlit.app/


## Deployment

To deploy this project run

```bash
  streamlit run app.py
```
python version-> 3.9-3.10

title: Text To Image
emoji: 💻
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.44.0
app_file: app.py
pinned: false
short_description: uses stable diffusion model
