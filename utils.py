import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def detect_landmarks_mediapipe(image_np):
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w = image_np.shape[:2]
        
        nose_indices = list(range(6, 18)) + list(range(164, 168)) + list(range(94, 100))
        nose_points = [(int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks.landmark) if i in nose_indices]

        return {
            "nose_tip": nose_points,
            "nose_bridge": nose_points[:4]
        }

def create_enhanced_nose_mask(image_array, landmarks, padding_factor=1.5, blur_radius=20):
    height, width = image_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    all_nose_points = landmarks["nose_tip"] + landmarks["nose_bridge"]
    x_coords = [p[0] for p in all_nose_points]
    y_coords = [p[1] for p in all_nose_points]

    min_x, max_x = max(0, min(x_coords)-30), min(width, max(x_coords)+30)
    min_y, max_y = max(0, min(y_coords)-30), min(height, max(y_coords)+30)

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    width_radius = (max_x - min_x) // 2 + 20
    height_radius = (max_y - min_y) // 2 + 25

    y, x = np.ogrid[:height, :width]
    ellipse_mask = ((x - center_x) ** 2 / width_radius ** 2 +
                    (y - center_y) ** 2 / height_radius ** 2) <= 1
    mask[ellipse_mask] = 255

    return cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
