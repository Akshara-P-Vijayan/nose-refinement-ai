import streamlit as st
from PIL import Image
import numpy as np
import torch
import face_recognition
from diffusers import StableDiffusionInpaintPipeline
import cv2

# === Nose mask creation ===
def create_enhanced_nose_mask(image_array, landmarks, padding_factor=1.5, blur_radius=20):
    height, width = image_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    nose_bridge = landmarks.get("nose_bridge", [])
    nose_tip = landmarks.get("nose_tip", [])
    if not nose_bridge or not nose_tip:
        return None
    all_nose_points = nose_bridge + nose_tip
    x_coords = [p[0] for p in all_nose_points]
    y_coords = [p[1] for p in all_nose_points]
    min_x, max_x = max(0, min(x_coords) - 30), min(width, max(x_coords) + 30)
    min_y, max_y = max(0, min(y_coords) - 30), min(height, max(y_coords) + 30)
    center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
    width_radius, height_radius = (max_x - min_x) // 2 + 20, (max_y - min_y) // 2 + 25
    y, x = np.ogrid[:height, :width]
    ellipse_mask = ((x - center_x)**2 / width_radius**2 + (y - center_y)**2 / height_radius**2) <= 1
    mask[ellipse_mask] = 255
    mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
    return Image.fromarray(mask)

# === Load Inpainting Pipeline ===
@st.cache_resource
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use float16 only if CUDA is available, otherwise float32
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        use_safetensors=True
    ).to(device)

    try:
        pipe.enable_attention_slicing()
        if device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        st.warning(f"Could not enable memory-efficient attention: {e}")

    return pipe


# === Streamlit UI ===
st.title("Nose Refinement Inpainting")
st.markdown("Upload a face image, enter prompts, and get a refined result using Stable Diffusion.")

uploaded_image = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
positive_prompt = st.text_area("Positive Prompt", 
    "refined nose, beautiful nose, realistic, elegant, natural skin, symmetrical")
negative_prompt = st.text_area("Negative Prompt", 
    "blurry, deformed, asymmetrical, wide nostrils, artificial, cartoon")

if uploaded_image and positive_prompt and negative_prompt:
    with st.spinner("Processing..."):
        input_image = Image.open(uploaded_image).convert("RGB").resize((512, 512))
        input_array = np.array(input_image)
        landmarks_list = face_recognition.face_landmarks(input_array)
        if not landmarks_list:
            st.error("No face landmarks found.")
        else:
            mask_image = create_enhanced_nose_mask(input_array, landmarks_list[0])
            if mask_image is None:
                st.error("Could not generate mask.")
            else:
                pipe = load_pipeline()
                generator = torch.Generator(device=pipe.device)  
                generator.manual_seed(42)
                result = pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    mask_image=mask_image,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    strength=0.75,
                    generator=generator
                ).images[0]
                
                st.image(result, caption="Refined Result", use_column_width=True)
                st.success("✅ Done!")

