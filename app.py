import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import torch
import cv2
from diffusers import StableDiffusionInpaintPipeline
import face_recognition
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings
import io
import tempfile
import os
import traceback

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="AI Nose Refinement Tool",
    page_icon="👃",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_pipeline_safely():
    """Load the inpainting pipeline with proper error handling and caching"""
    try:
        with st.spinner("🔄 Loading AI model (this may take a moment)..."):
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.info(f"Using device: {device}")
            
            # Use CPU-friendly settings if no CUDA
            if device == "cpu":
                torch_dtype = torch.float32
                st.warning("⚠️ Running on CPU - this will be slower")
            else:
                torch_dtype = torch.float16
            
            # Try loading with error handling
            try:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    safety_checker=None,  # Disable safety checker to reduce memory usage
                    requires_safety_checker=False
                )
            except Exception as e:
                st.warning(f"Primary model failed: {e}")
                # Try alternative model
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            # Move to device
            pipe = pipe.to(device)
            
            # Enable memory-efficient attention if on CPU
            if device == "cpu":
                try:
                    pipe.enable_attention_slicing(1)
                    st.success("✅ Enabled attention slicing for CPU")
                except:
                    pass
            else:
                # GPU optimizations
                try:
                    pipe.enable_model_cpu_offload()
                    st.success("✅ Enabled CPU offload")
                except:
                    pass
                
                try:
                    pipe.enable_attention_slicing(1)
                    st.success("✅ Enabled attention slicing")
                except:
                    pass
            
            st.success("🎉 Model loaded successfully!")
            return pipe
            
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.error(f"Error details: {traceback.format_exc()}")
        raise

def create_enhanced_nose_mask(image_array, landmarks, padding_factor=1.5, blur_radius=20):
    """Create a more comprehensive nose mask with better coverage"""
    try:
        height, width = image_array.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get nose landmarks with expanded coverage
        nose_bridge = landmarks.get("nose_bridge", [])
        nose_tip = landmarks.get("nose_tip", [])
        
        if not nose_bridge and not nose_tip:
            raise ValueError("No nose landmarks found")
        
        # Combine all nose points
        all_nose_points = nose_bridge + nose_tip
        
        # Create a more generous bounding area around the nose
        if all_nose_points:
            # Get bounding box of nose
            x_coords = [p[0] for p in all_nose_points]
            y_coords = [p[1] for p in all_nose_points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Expand the bounding box
            expansion = 30  # pixels
            min_x = max(0, min_x - expansion)
            max_x = min(width, max_x + expansion)
            min_y = max(0, min_y - expansion)
            max_y = min(height, max_y + expansion)
            
            # Create elliptical mask around nose area
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            width_radius = (max_x - min_x) // 2 + 20
            height_radius = (max_y - min_y) // 2 + 25
            
            # Draw filled ellipse
            y, x = np.ogrid[:height, :width]
            ellipse_mask = ((x - center_x) ** 2 / width_radius ** 2 + 
                           (y - center_y) ** 2 / height_radius ** 2) <= 1
            mask[ellipse_mask] = 255
        
        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
        
        return mask
    except Exception as e:
        st.error(f"Error creating mask: {str(e)}")
        raise

def refine_nose(input_image, positive_prompt, negative_prompt, strength=0.75, num_inference_steps=30, guidance_scale=7.5, seed=42):
    """Main nose refinement function"""
    try:
        # Resize image to 512x512 for the model
        input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
        input_array = np.array(input_image)
        
        # Detect landmarks
        st.info("🔍 Detecting face landmarks...")
        landmarks_list = face_recognition.face_landmarks(input_array)
        if not landmarks_list:
            raise ValueError("❌ No face landmarks detected. Please ensure the image contains a clear face.")
        
        landmarks = landmarks_list[0]
        st.success("✅ Face landmarks detected!")
        
        # Create enhanced mask
        st.info("🎭 Creating processing mask...")
        mask_array = create_enhanced_nose_mask(input_array, landmarks)
        mask = Image.fromarray(mask_array)
        st.success("✅ Mask created!")
        
        # Load pipeline
        st.info("🤖 Loading AI model...")
        pipe = load_pipeline_safely()
        
        # Set up generator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate refined nose
        st.info("🎨 Generating refined image...")
        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator
        ).images[0]
        
        st.success("✅ Image processing complete!")
        return result, mask
        
    except Exception as e:
        st.error(f"❌ Error in refinement: {str(e)}")
        st.error(f"Error details: {traceback.format_exc()}")
        raise

# Streamlit UI
def main():
    try:
        st.title("🎭 AI Nose Refinement Tool")
        st.markdown("Transform facial features using advanced AI technology")
        
        # System info
        with st.expander("🔧 System Information"):
            st.write(f"PyTorch version: {torch.__version__}")
            st.write(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.write(f"CUDA device: {torch.cuda.get_device_name()}")
            st.write(f"OpenCV version: {cv2.__version__}")
        
        # Sidebar for parameters
        st.sidebar.header("⚙️ Settings")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "📸 Upload an image", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear portrait image for nose refinement"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                input_image = Image.open(uploaded_file).convert("RGB")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📤 Original Image")
                    st.image(input_image, use_column_width=True)
                
                # Sidebar parameters
                st.sidebar.subheader("🎨 Prompt Settings")
                
                # Default prompts
                default_positive = (
                    "beautiful refined nose, straight nasal bridge, elegant nose tip, "
                    "natural proportions, smooth skin, perfect symmetry, "
                    "professional photography, high quality, detailed, realistic"
                )
                
                default_negative = (
                    "deformed, ugly, blurry, low quality, distorted, asymmetrical, "
                    "crooked nose, bulbous tip, wide nostrils, plastic, artificial, "
                    "cartoon, painting, sketch"
                )
                
                positive_prompt = st.sidebar.text_area(
                    "✨ Positive Prompt",
                    value=default_positive,
                    height=100,
                    help="Describe the desired nose features"
                )
                
                negative_prompt = st.sidebar.text_area(
                    "🚫 Negative Prompt",
                    value=default_negative,
                    height=100,
                    help="Describe what to avoid"
                )
                
                st.sidebar.subheader("🔧 Model Parameters")
                
                strength = st.sidebar.slider(
                    "Transformation Strength", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.75, 
                    step=0.05,
                    help="How much to change the original image (higher = more dramatic)"
                )
                
                num_inference_steps = st.sidebar.slider(
                    "Quality Steps", 
                    min_value=10, 
                    max_value=50, 
                    value=20,  # Reduced for faster processing
                    step=5,
                    help="More steps = higher quality but slower processing"
                )
                
                guidance_scale = st.sidebar.slider(
                    "Prompt Adherence", 
                    min_value=1.0, 
                    max_value=20.0, 
                    value=7.5, 
                    step=0.5,
                    help="How closely to follow the prompt"
                )
                
                seed = st.sidebar.number_input(
                    "🌱 Seed", 
                    min_value=0, 
                    max_value=999999, 
                    value=42,
                    help="For reproducible results"
                )
                
                # Refinement type
                refinement_type = st.sidebar.selectbox(
                    "🎯 Refinement Type",
                    ["Subtle", "Moderate", "Dramatic"],
                    index=1
                )
                
                # Adjust strength based on refinement type
                if refinement_type == "Subtle":
                    strength = min(strength, 0.6)
                elif refinement_type == "Dramatic":
                    strength = max(strength, 0.8)
                
                # Process button
                if st.sidebar.button("🚀 Start Refinement", type="primary"):
                    try:
                        with st.spinner("🔄 Processing image... This may take a few minutes."):
                            result_image, mask_image = refine_nose(
                                input_image, 
                                positive_prompt, 
                                negative_prompt, 
                                strength, 
                                num_inference_steps, 
                                guidance_scale, 
                                seed
                            )
                        
                        with col2:
                            st.subheader("✨ Refined Image")
                            st.image(result_image, use_column_width=True)
                        
                        # Show mask
                        st.subheader("🎭 Processing Mask")
                        st.image(mask_image, use_column_width=True, caption="Areas modified by AI")
                        
                        # Download button
                        buf = io.BytesIO()
                        result_image.save(buf, format='PNG')
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="📥 Download Refined Image",
                            data=byte_im,
                            file_name="refined_nose.png",
                            mime="image/png"
                        )
                        
                        st.success("🎉 Nose refinement completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error occurred: {str(e)}")
                        st.error(f"Full error: {traceback.format_exc()}")
                        st.info("💡 Tips: Ensure your image contains a clear, front-facing face with good lighting.")
            
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
        
        else:
            st.info("👆 Please upload an image to get started!")
            
            # Show example
            st.subheader("📋 How to use:")
            st.markdown("""
            1. **Upload Image**: Choose a clear portrait photo
            2. **Customize Prompts**: Describe desired nose features
            3. **Adjust Settings**: Fine-tune the transformation
            4. **Process**: Click 'Start Refinement' and wait
            5. **Download**: Save your refined image
            
            **Tips for best results:**
            - Use high-quality, well-lit photos
            - Ensure the face is clearly visible and front-facing
            - Experiment with different prompt descriptions
            - Start with subtle changes before trying dramatic ones
            """)
    
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.error(f"Full error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
