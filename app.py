import sys
import streamlit as st
import torch
from PIL import Image
import cv2
import diffusers
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")

# Version info
st.write("### Environment & Library Versions:")
st.write(f"Python executable: `{sys.executable}`")
st.write(f"Python version: `{sys.version.split()[0]}`")
st.write(f"PyTorch version: `{torch.__version__}`")
st.write(f"CUDA available: `{torch.cuda.is_available()}`")
if torch.cuda.is_available():
    st.write(f"CUDA version: `{torch.version.cuda}`")
st.write(f"Pillow (PIL) version: `{Image.__version__}`")
st.write(f"OpenCV version: `{cv2.__version__}`")
st.write(f"Diffusers version: `{diffusers.__version__}`")
st.write(f"face_recognition version: `{face_recognition.__version__}`")
st.write(f"NumPy version: `{np.__version__}`")
st.write(f"Matplotlib version: `{matplotlib.__version__}`")

def create_enhanced_nose_mask(image_array, landmarks, padding_factor=1.5, blur_radius=30):
    height, width = image_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    nose_bridge = landmarks.get("nose_bridge", [])
    nose_tip = landmarks.get("nose_tip", [])
    
    if not nose_bridge or not nose_tip:
        raise ValueError("Insufficient nose landmarks")
    
    all_nose_points = nose_bridge + nose_tip
    
    x_coords = [p[0] for p in all_nose_points]
    y_coords = [p[1] for p in all_nose_points]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    expansion = 30
    min_x = max(0, min_x - expansion)
    max_x = min(width, max_x + expansion)
    min_y = max(0, min_y - expansion)
    max_y = min(height, max_y + expansion)
    
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    width_radius = (max_x - min_x) // 2 + 20
    height_radius = (max_y - min_y) // 2 + 25
    
    y, x = np.ogrid[:height, :width]
    ellipse_mask = ((x - center_x) ** 2 / width_radius ** 2 + 
                   (y - center_y) ** 2 / height_radius ** 2) <= 1
    mask[ellipse_mask] = 255
    
    # Blur the edges for smooth transition
    mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
    
    return mask

@st.cache_resource(show_spinner=False)
def load_pipeline():
    st.info("Loading AI model (this may take a moment)...")
    try:
        pipe = diffusers.StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        # Optimizations
        try:
            pipe.enable_attention_slicing()
        except: pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except: pass
        try:
            pipe.enable_model_cpu_offload()
        except: pass
        st.success("Model loaded successfully!")
        return pipe
    except Exception as e:
        st.error(f"Failed to load primary model: {e}")
        st.info("Trying fallback model...")
        try:
            pipe = diffusers.StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16
            )
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            st.success("Fallback model loaded successfully!")
            return pipe
        except Exception as e2:
            st.error(f"Fallback model failed too: {e2}")
            return None

def refine_nose_dramatic(input_image, positive_prompt, negative_prompt, pipe):
    original_image = input_image.convert("RGB")
    original_size = original_image.size
    
    input_image_resized = original_image.resize((512, 512), Image.Resampling.LANCZOS)
    input_array = np.array(input_image_resized)
    
    landmarks_list = face_recognition.face_landmarks(input_array)
    if not landmarks_list:
        raise ValueError("No face landmarks detected in the image.")
    landmarks = landmarks_list[0]
    
    mask_array = create_enhanced_nose_mask(input_array, landmarks, blur_radius=30)
    mask = Image.fromarray(mask_array)
    
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
    
    result = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=input_image_resized,
        mask_image=mask,
        num_inference_steps=40,
        guidance_scale=8.5,
        strength=0.85,
        generator=generator
    ).images[0]
    
    result = result.resize(original_size, Image.Resampling.LANCZOS)
    return original_image, result

def main():
    st.title("Nose Refinement with Stable Diffusion Inpainting")
    
    st.markdown("""
    Upload a face image and enter your own **Dramatic Refinement** prompts:
    - Positive prompt (what you want to enhance)
    - Negative prompt (what you want to avoid)
    """)
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    positive_prompt = st.text_area("Positive prompt", 
        value="perfectly refined nose, sleek nasal bridge, elegant pointed tip, balanced nostril size, smooth contours, ideal facial harmony, professional portrait lighting, crisp details, natural skin, symmetrical features, refined beauty, high resolution"
    )
    negative_prompt = st.text_area("Negative prompt",
        value="wide nose, round tip, thick bridge, large nostrils, crooked, bumpy surface, irregular shape, unnatural, fake looking, low quality, blurry, distorted"
    )
    
    pipe = load_pipeline()
    
    if not pipe:
        st.error("Model pipeline not available, cannot proceed.")
        return
    
    if uploaded_file:
        try:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Run Dramatic Nose Refinement"):
                with st.spinner("Refining nose..."):
                    original, refined = refine_nose_dramatic(input_image, positive_prompt, negative_prompt, pipe)
                
                st.success("Refinement completed!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original, caption="Before", use_column_width=True)
                with col2:
                    st.image(refined, caption="After", use_column_width=True)
                
                # Allow user to download result
                buf = st.experimental_bytes_io()
                refined.save(buf, format="PNG")
                st.download_button("Download Refined Image", data=buf.getvalue(), file_name="nose_refined.png", mime="image/png")
        
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
