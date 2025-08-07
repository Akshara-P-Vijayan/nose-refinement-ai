import streamlit as st
import os
import sys

# Check if running on Streamlit Cloud and handle imports gracefully
try:
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from scipy import ndimage
    import warnings
    import io
    import tempfile
    import traceback
    
    # Try importing ML libraries
    ML_AVAILABLE = True
    try:
        import torch
        import mediapipe as mp
        from diffusers import StableDiffusionInpaintPipeline
        
        # Try face-recognition, fallback to mediapipe
        try:
            import face_recognition
            FACE_DETECTION_METHOD = "face_recognition"
        except ImportError:
            FACE_DETECTION_METHOD = "mediapipe"
            
    except ImportError as e:
        ML_AVAILABLE = False
        st.error(f"ML libraries not available: {e}")
        
except ImportError as e:
    st.error(f"Basic libraries not available: {e}")
    st.stop()

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="AI Nose Refinement Tool",
    page_icon="👃",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_environment():
    """Check if the environment supports the app"""
    issues = []
    
    if not ML_AVAILABLE:
        issues.append("PyTorch and related ML libraries are not available")
    
    # Check device capabilities
    if ML_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            issues.append("Running on CPU - processing will be very slow")
    
    return issues

def create_simple_nose_mask_mediapipe(image_array):
    """Create nose mask using MediaPipe (lighter alternative)"""
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    height, width = image_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        results = face_mesh.process(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Nose landmark indices for MediaPipe
            nose_indices = [1, 2, 5, 6, 19, 20, 94, 95, 125, 126, 131, 132, 134, 135, 
                          142, 143, 182, 183, 193, 194, 195, 196, 197, 198, 236, 237, 
                          238, 239, 240, 241, 242, 245, 246, 247, 248, 249, 250, 251, 
                          252, 253, 254, 255, 256, 257, 258, 259, 260, 294, 295, 296, 
                          297, 298, 299, 300, 305, 306, 307, 308, 309, 310, 311, 312, 
                          313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 
                          325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 
                          337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 
                          349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 
                          361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 
                          373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 
                          385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 
                          397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 
                          409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 
                          421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 
                          433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 
                          445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 
                          457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468]
            
            # Simple nose area (just key points)
            key_nose_indices = [1, 2, 5, 6, 19, 20, 125, 131, 134, 142, 236, 237, 238, 239, 240, 241, 242]
            
            nose_points = []
            for idx in key_nose_indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    nose_points.append((x, y))
            
            if nose_points:
                # Create bounding box around nose
                x_coords = [p[0] for p in nose_points]
                y_coords = [p[1] for p in nose_points]
                
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                # Expand bounding box
                expansion = 40
                min_x = max(0, min_x - expansion)
                max_x = min(width, max_x + expansion)
                min_y = max(0, min_y - expansion)
                max_y = min(height, max_y + expansion)
                
                # Create elliptical mask
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2
                width_radius = (max_x - min_x) // 2 + 10
                height_radius = (max_y - min_y) // 2 + 15
                
                y, x = np.ogrid[:height, :width]
                ellipse_mask = ((x - center_x) ** 2 / width_radius ** 2 + 
                               (y - center_y) ** 2 / height_radius ** 2) <= 1
                mask[ellipse_mask] = 255
    
    # Apply blur
    mask = cv2.GaussianBlur(mask, (41, 41), 15)
    return mask

def create_enhanced_nose_mask_face_recognition(image_array, landmarks, blur_radius=20):
    """Original face-recognition based mask creation"""
    height, width = image_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Get nose landmarks
    nose_bridge = landmarks.get("nose_bridge", [])
    nose_tip = landmarks.get("nose_tip", [])
    
    if not nose_bridge and not nose_tip:
        raise ValueError("No nose landmarks found")
    
    all_nose_points = nose_bridge + nose_tip
    
    if all_nose_points:
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
    
    mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
    return mask

@st.cache_resource
def load_pipeline_safely():
    """Load the inpainting pipeline with comprehensive error handling"""
    if not ML_AVAILABLE:
        raise RuntimeError("ML libraries not available")
        
    try:
        with st.spinner("🔄 Loading AI model..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float32 if device == "cpu" else torch.float16
            
            st.info(f"Using device: {device}")
            
            # Try primary model
            try:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            except Exception as e:
                st.warning(f"Primary model failed, trying fallback: {e}")
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            pipe = pipe.to(device)
            
            # Optimizations
            if device == "cpu":
                pipe.enable_attention_slicing(1)
            else:
                try:
                    pipe.enable_model_cpu_offload()
                    pipe.enable_attention_slicing(1)
                except:
                    pass
            
            st.success("✅ Model loaded successfully!")
            return pipe
            
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        raise

def refine_nose(input_image, positive_prompt, negative_prompt, strength=0.75, num_inference_steps=20, guidance_scale=7.5, seed=42):
    """Main nose refinement function with fallbacks"""
    
    if not ML_AVAILABLE:
        raise RuntimeError("ML libraries not available")
    
    try:
        # Resize image
        input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
        input_array = np.array(input_image)
        
        # Detect face and create mask
        st.info("🔍 Detecting face landmarks...")
        
        if FACE_DETECTION_METHOD == "face_recognition":
            landmarks_list = face_recognition.face_landmarks(input_array)
            if not landmarks_list:
                raise ValueError("No face detected with face_recognition")
            landmarks = landmarks_list[0]
            mask_array = create_enhanced_nose_mask_face_recognition(input_array, landmarks)
        else:
            # Use MediaPipe fallback
            mask_array = create_simple_nose_mask_mediapipe(input_array)
            if mask_array is None or np.sum(mask_array) == 0:
                raise ValueError("No face detected with MediaPipe")
        
        mask = Image.fromarray(mask_array)
        st.success("✅ Face landmarks detected!")
        
        # Load and run pipeline
        pipe = load_pipeline_safely()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)
        
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
        
        st.success("✅ Processing complete!")
        return result, mask
        
    except Exception as e:
        st.error(f"❌ Error in refinement: {str(e)}")
        raise

def main():
    st.title("🎭 AI Nose Refinement Tool")
    st.markdown("Transform facial features using advanced AI technology")
    
    # Environment check
    issues = check_environment()
    if issues:
        st.warning("⚠️ Environment Issues:")
        for issue in issues:
            st.write(f"• {issue}")
        
        if not ML_AVAILABLE:
            st.error("❌ Cannot run without ML libraries. Please check your requirements.txt")
            st.info("💡 Make sure you have the correct PyTorch installation for your platform")
            st.stop()
    
    # System info
    with st.expander("🔧 System Information"):
        if ML_AVAILABLE:
            st.write(f"PyTorch version: {torch.__version__}")
            st.write(f"CUDA available: {torch.cuda.is_available()}")
            st.write(f"Face detection: {FACE_DETECTION_METHOD}")
        st.write(f"OpenCV version: {cv2.__version__}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📸 Upload an image", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear portrait image"
    )
    
    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Original Image")
                st.image(input_image, use_column_width=True)
            
            # Sidebar parameters
            st.sidebar.header("⚙️ Settings")
            
            positive_prompt = st.sidebar.text_area(
                "✨ Positive Prompt",
                value="beautiful refined nose, straight nasal bridge, elegant nose tip, natural proportions, smooth skin, perfect symmetry, professional photography, high quality, detailed, realistic",
                height=100
            )
            
            negative_prompt = st.sidebar.text_area(
                "🚫 Negative Prompt", 
                value="deformed, ugly, blurry, low quality, distorted, asymmetrical, crooked nose, bulbous tip, wide nostrils, plastic, artificial, cartoon, painting, sketch",
                height=100
            )
            
            strength = st.sidebar.slider("Transformation Strength", 0.1, 1.0, 0.75, 0.05)
            num_inference_steps = st.sidebar.slider("Quality Steps", 10, 30, 15, 5)
            guidance_scale = st.sidebar.slider("Prompt Adherence", 1.0, 20.0, 7.5, 0.5)
            seed = st.sidebar.number_input("🌱 Seed", 0, 999999, 42)
            
            if st.sidebar.button("🚀 Start Refinement", type="primary"):
                if not ML_AVAILABLE:
                    st.error("❌ ML libraries not available")
                    return
                
                try:
                    with st.spinner("🔄 Processing... This may take several minutes on CPU."):
                        result_image, mask_image = refine_nose(
                            input_image, positive_prompt, negative_prompt,
                            strength, num_inference_steps, guidance_scale, seed
                        )
                    
                    with col2:
                        st.subheader("✨ Refined Image")
                        st.image(result_image, use_column_width=True)
                    
                    st.subheader("🎭 Processing Mask")
                    st.image(mask_image, use_column_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    result_image.save(buf, format='PNG')
                    st.download_button(
                        "📥 Download Result",
                        buf.getvalue(),
                        "refined_nose.png",
                        "image/png"
                    )
                    
                    st.success("🎉 Complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
    
    else:
        st.info("👆 Please upload an image to get started!")

if __name__ == "__main__":
    main()
