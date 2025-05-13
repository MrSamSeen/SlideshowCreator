import cv2
import mediapipe as mp
from mediapipe.python.solutions.face_detection import FaceDetection
from functools import lru_cache
import os

from .config import CACHE_SIZE
from .utils import print_error, print_info, print_warning

# Initialize face detection model once for reuse
face_detection_model = None

# Use LRU cache to avoid re-detecting faces in the same image
@lru_cache(maxsize=CACHE_SIZE)
def detect_face_cached(image_path):
    """Cached version of face detection that works with file paths"""
    global face_detection_model

    # Load the image
    print_info(f"[FACE_DETECT] Attempting to load image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print_error(f"[FACE_DETECT] Failed to load image: {image_path}")
        # Return a default center if image loading fails, including original shape placeholder
        # This requires knowing a default shape or handling it upstream.
        # For now, returning None and letting caller handle it might be safer.
        return None # Or a default like (0,0,0, (0,0))

    # Initialize the model if not already done
    if face_detection_model is None:
        face_detection_model = FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

    # Convert the BGR image to RGB and process it
    print_info(f"[FACE_DETECT] Image {os.path.basename(image_path)} loaded successfully. Shape: {image.shape}")
    print_info(f"[FACE_DETECT] Processing image {os.path.basename(image_path)} for face detection...")
    results = face_detection_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print_info(f"[FACE_DETECT] Raw detection results for {os.path.basename(image_path)}: {results}")

    image_shape = image.shape # Store for return
    h, w = image_shape[:2]

    # If results object itself is None (e.g., error in processing)
    if results is None:
        print_warning(f"[FACE_DETECT] Face detection processing returned None for image: {os.path.basename(image_path)}. Defaulting to center.")
        return (w // 2, h // 2, w // 4, image_shape)  # Default: center

    # Try to get detections attribute; it might be None if no faces are found
    detections = getattr(results, 'detections', None)
    print_info(f"[FACE_DETECT] Detections attribute for {os.path.basename(image_path)}: {detections}")

    # If no detections are found (either attribute missing or detections list is empty/None)
    if not detections:
        print_warning(f"[FACE_DETECT] No faces detected in {os.path.basename(image_path)}. Defaulting to center.")
        return (w // 2, h // 2, w // 4, image_shape)  # Default: center

    # Get the first (presumably main) face
    detection = detections[0]

    # Get bounding box
    bboxC = detection.location_data.relative_bounding_box
    
    # Convert relative coordinates to absolute
    x = int(bboxC.xmin * w)
    y = int(bboxC.ymin * h)
    width = int(bboxC.width * w)
    height = int(bboxC.height * h)

    # Calculate center of the face
    center_x = x + width // 2
    center_y = y + height // 2

    # Use the larger dimension (width or height) as the size
    size = max(width, height)

    print_info(f"[FACE_DETECT] Detected face in {os.path.basename(image_path)} at ({center_x}, {center_y}) with size {size}. Original shape: {image_shape}")
    return (center_x, center_y, size, image_shape)