import streamlit as st
import numpy as np
from PIL import Image
import torch
import cv2
from diffusers import StableDiffusionInpaintPipeline
from transformers import DPTImageProcessor, DPTForDepthEstimation
import matplotlib.pyplot as plt
import tempfile
from utils import create_enhanced_nose_mask, detect_landmarks_mediapipe  
import os

@st.cache_resource
def load_inpaint_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()
    return pipe

pipe = load_inpaint_pipeline()


st.title("👃 AI Nose Refinement (Subtle/Dramatic)")
st.markdown("Upload your image and customize prompts to enhance your nose using Stable Diffusion inpainting.")

input_image = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
refine_type = st.radio("Choose Refinement Type", ["Subtle", "Dramatic"])
positive_prompt = st.text_area("Positive Prompt", "refined nose, smooth bridge, elegant tip, photorealistic, high quality")
negative_prompt = st.text_area("Negative Prompt", "blurry, deformed, cartoonish, unnatural nose")

if st.button("🔧 Run Refinement") and input_image:
    with st.spinner("Processing..."):
        
        image = Image.open(input_image).convert("RGB").resize((512, 512))
        image_np = np.array(image)

     
        landmarks = detect_landmarks_mediapipe(image_np)

        if not landmarks:
            st.error("Could not detect facial landmarks.")
        else:
            
            mask = create_enhanced_nose_mask(image_np, landmarks, blur_radius=20 if refine_type == "Subtle" else 30)
            mask_pil = Image.fromarray(mask)

            
            generator = torch.Generator(device=pipe.device).manual_seed(42)

            result = pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask_pil,
                num_inference_steps=30 if refine_type == "Subtle" else 40,
                guidance_scale=7.5 if refine_type == "Subtle" else 8.5,
                strength=0.75 if refine_type == "Subtle" else 0.85,
                generator=generator
            ).images[0]

            st.success("Refinement complete!")
            st.image(result, caption="Refined Nose Output", use_column_width=True)
            st.download_button("📥 Download Image", data=result.tobytes(), file_name="refined_nose.png", mime="image/png")
