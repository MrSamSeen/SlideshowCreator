import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from .config import (
    HOLD_FRAMES, TOTAL_FRAMES, ZOOM_AMOUNT, 
    BLUR_STRENGTH, STRETCH_AMOUNT, BLUR_FACE_PROTECTION, NUM_WORKERS
)
from .effects import apply_fisheye_effect, apply_radial_blur

def process_frame(args):
    """Process a single frame for the transition - used with multiprocessing"""
    img, face_center, zoom_progress, effect_progress, is_first_image = args

    if img is None or face_center is None:
        # Handle cases where image or face_center might be None (e.g., failed to load/detect)
        # This might involve returning a black frame or a copy of a default image
        # For now, let's assume img and face_center are valid if they reach here
        # Or, raise an error / log a warning
        print(f"Error: process_frame received None for img or face_center.")
        # Fallback: return a black frame of a standard size if possible, or handle upstream
        # This part needs careful consideration based on how errors are propagated.
        return np.zeros((1080, 1920, 3), dtype=np.uint8) # Example fallback

    h, w = img.shape[:2]
    cx, cy, face_size = face_center[:3]

    # For zoom effect, we'll use a transformation matrix that scales from the face center
    # Zoom logic:
    # - First image: Zooms IN towards the face.
    # - Second image: Starts zoomed IN on the face, then zooms OUT.
    if is_first_image:
        # For first image, we zoom IN toward the face
        # Start at 1.0 (no zoom) and go to ZOOM_AMOUNT (zoomed in)
        scale = 1.0 + (ZOOM_AMOUNT - 1.0) * zoom_progress
    else:
        # For second image, we start zoomed IN and zoom OUT from the face
        # Start at ZOOM_AMOUNT (zoomed in) and go to 1.0 (no zoom)
        scale = ZOOM_AMOUNT - (ZOOM_AMOUNT - 1.0) * zoom_progress

    # Create the zoom transformation matrix
    # This matrix will scale the image from the face center
    M = np.array([
        [scale, 0, cx * (1 - scale)],
        [0, scale, cy * (1 - scale)]
    ], dtype=np.float32)

    # Apply the zoom transformation
    result = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4)

    # Adjust face center and size based on zoom
    # The face center stays the same, but the size changes
    scaled_face_size = int(face_size * scale)
    scaled_face_center = (cx, cy, scaled_face_size)

    # Apply fisheye/stretching effect (stronger toward the middle of transition)
    stretch_strength = STRETCH_AMOUNT * effect_progress
    if stretch_strength > 0.01:
        result = apply_fisheye_effect(
            result,
            scaled_face_center,
            strength=stretch_strength
        )

    # Apply radial blur (stronger in the middle of the transition)
    blur_strength_val = int(BLUR_STRENGTH * effect_progress)
    if blur_strength_val > 0:
        result = apply_radial_blur(
            result,
            scaled_face_center,
            strength=blur_strength_val,
            face_protection=BLUR_FACE_PROTECTION
        )

    return result

def create_zoom_transition(img1, img2, face_center1, face_center2):
    """Create a zoom transition between two images with fisheye effect.

    The transition zooms in toward the face in the first image.
    Then, it switches to the second image, which starts zoomed in on its face
    and then zooms out to reveal the full second image.

    Args:
        img1: First image (NumPy array)
        img2: Second image (NumPy array)
        face_center1: (x, y, size) of the face in the first image
        face_center2: (x, y, size) of the face in the second image

    Returns:
        List of frames for the transition
    """
    frames = []

    if img1 is None or img2 is None or face_center1 is None or face_center2 is None:
        print("Error: create_zoom_transition received None for one of its inputs.")
        # Return empty list or handle error appropriately
        return frames

    # Hold the first image
    first_frame = img1.copy()
    frames.extend([first_frame] * HOLD_FRAMES)

    # Calculate transition frames (excluding hold frames)
    num_transition_frames = TOTAL_FRAMES - 2 * HOLD_FRAMES
    if num_transition_frames <= 0:
        # If no transition frames (e.g., HOLD_DURATION is too long relative to TRANSITION_DURATION)
        # just hold img1, then img2
        last_frame = img2.copy()
        frames.extend([last_frame] * HOLD_FRAMES)
        return frames

    # Prepare arguments for parallel processing
    frame_args = []

    # Create transition frames
    for i in range(num_transition_frames):
        # Calculate progress (0 to 1)
        progress = i / (num_transition_frames - 1) if num_transition_frames > 1 else 0.5

        # Calculate effect strength based on transition progress
        # Effect is strongest in the middle of the transition
        effect_progress = 1 - abs(2 * progress - 1)

        # First half of transition - zoom in toward face in first image
        if progress < 0.5:
            # Normalize progress for first half (0 to 1)
            zoom_progress = progress * 2
            frame_args.append((img1, face_center1, zoom_progress, effect_progress, True))
        # Second half of transition - zoom in toward face in second image
        else:
            # Normalize progress for second half (0 to 1)
            zoom_progress = (progress - 0.5) * 2
            frame_args.append((img2, face_center2, zoom_progress, effect_progress, False))

    # Process frames
    processed_transition_frames = []
    if frame_args: # Ensure there are arguments to process
        if NUM_WORKERS > 1:
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                processed_transition_frames = list(executor.map(process_frame, frame_args))
        else:
            processed_transition_frames = [process_frame(args) for args in frame_args]

    # Add transition frames to the result
    frames.extend(processed_transition_frames)

    # Hold the second image
    last_frame = img2.copy()
    frames.extend([last_frame] * HOLD_FRAMES)

    return frames