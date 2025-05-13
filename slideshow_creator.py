#!/usr/bin/env python3
"""
Slideshow Creator - Creates smooth zoom transitions between images with face detection.

This script creates a video slideshow from a collection of images with smooth zoom transitions
that focus on faces. It detects faces in each image, creates transitions by zooming out from
the first image and zooming in to the second image, applies special effects during transitions,
and combines all frames into a high-quality video with optional background music.
"""

import os
import cv2
import glob
import numpy as np
import mediapipe as mp
import datetime
import math
import random
import time
import shutil
import mediapipe as mp
from tqdm import tqdm
import multiprocessing
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

# ===== CONFIGURATION PARAMETERS =====

# Input/Output Settings
INPUT_DIR = "input"           # Directory containing input images
OUTPUT_DIR = "output"         # Directory for output files
DONE_DIR = "done"             # Directory to move processed images
MUSIC_DIR = "music"           # Directory containing background music files

# Processing Settings
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Use all but one CPU core
MIN_IMAGES = 30               # Minimum number of images required to start a batch
WAIT_TIME = 60                # Time to wait between checks or batches (in seconds)

# Transition Settings
FPS = 30                      # Frames per second for the output video
TRANSITION_DURATION = 2.4     # Duration of each transition in seconds
HOLD_DURATION = 0.6           # How long to hold each image before/after transition
TOTAL_FRAMES = int(TRANSITION_DURATION * FPS)  # Total frames per transition
HOLD_FRAMES = int(HOLD_DURATION * FPS)         # Frames to hold each image

# Effect Parameters
ZOOM_AMOUNT = 2.0             # Maximum zoom factor (2.0 = zoom in/out by 2x)
BLUR_STRENGTH = 15            # Maximum blur strength during transitions
BLUR_FACE_PROTECTION = 1.0    # How much to protect faces from blur (0-1)
STRETCH_AMOUNT = 0.3          # Fisheye/stretch effect strength
STRETCH_FACE_PROTECTION = 1.0 # How much to protect faces from stretching

# Video Output Parameters
VIDEO_QUALITY = 20            # CRF value for video compression (lower = higher quality)
VIDEO_RESOLUTION = (1920, 1080)  # Output resolution (width, height) - 1080p by default
CREATE_COMPARISON = True      # Whether to create side-by-side comparison frames
COMPARISON_INTERVAL = 10      # Save comparison every N frames
ADD_BACKGROUND_MUSIC = True   # Whether to add background music to the video

# Performance Parameters
JPEG_QUALITY = 100             # JPEG quality for saving frames (0-100)
USE_FAST_BLUR = True          # Use faster blur algorithm for preview frames
CACHE_SIZE = 100              # Size of LRU cache for face detection

# ===== TERMINAL FORMATTING =====

class Colors:
    """Terminal colors for formatted output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a formatted header"""
    terminal_width = shutil.get_terminal_size().columns
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * terminal_width}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(terminal_width)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * terminal_width}{Colors.ENDC}\n")

def print_success(text):
    """Print a success message"""
    print(f"{Colors.GREEN}{Colors.BOLD}✓ {text}{Colors.ENDC}")

def print_info(text):
    """Print an info message"""
    print(f"{Colors.YELLOW}ℹ {text}{Colors.ENDC}")

def print_warning(text):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_error(text):
    """Print an error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_waiting(text, seconds):
    """Print a waiting message with countdown"""
    for i in range(seconds, 0, -1):
        print(f"{Colors.CYAN}⏱ {text} ({i}s remaining)...{Colors.ENDC}", end='\r')
        time.sleep(1)
    print(" " * shutil.get_terminal_size().columns, end='\r')  # Clear the line

# ===== FACE DETECTION =====

# MediaPipe setup for face detection
from mediapipe.python.solutions.face_detection import FaceDetection

# Initialize face detection model once for reuse
face_detection_model = None

# Use LRU cache to avoid re-detecting faces in the same image
@lru_cache(maxsize=CACHE_SIZE)
def detect_face_cached(image_path):
    """Cached version of face detection that works with file paths"""
    global face_detection_model

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print_error(f"Failed to load image: {image_path}")
        return None

    # Initialize the model if not already done
    if face_detection_model is None:
        face_detection_model = FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

    # Convert the BGR image to RGB and process it
    results = face_detection_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # If results object itself is None (e.g., error in processing)
    if results is None:
        h, w = image.shape[:2]
        # print_warning(f"Face detection processing returned None for image: {image_path}") # Optional
        return (w // 2, h // 2, w // 4, image.shape)  # Default: center

    # Try to get detections attribute; it might be None if no faces are found
    detections = getattr(results, 'detections', None)

    # If no detections are found (either attribute missing or detections list is empty/None)
    if not detections:
        h, w = image.shape[:2]
        return (w // 2, h // 2, w // 4, image.shape)  # Default: center

    # Get the first (presumably main) face
    detection = detections[0]

    # Get bounding box
    bbox = detection.location_data.relative_bounding_box
    h, w = image.shape[:2]

    # Convert relative coordinates to absolute
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    width = int(bbox.width * w)
    height = int(bbox.height * h)

    # Calculate center of the face
    center_x = x + width // 2
    center_y = y + height // 2

    # Use the larger dimension (width or height) as the size
    size = max(width, height)

    return (center_x, center_y, size, image.shape)

# ===== SPECIAL EFFECTS =====

def apply_fisheye_effect(image, face_center, strength=0.3):
    """Apply fisheye/stretching effect that stretches toward edges while protecting face area"""
    # If strength is too low, don't apply any effect
    if strength < 0.001:
        return image.copy()

    h, w = image.shape[:2]
    cx, cy, face_size = face_center[:3]

    # Create coordinate grid
    y, x = np.mgrid[0:h, 0:w]

    # Calculate distance from face center for each pixel
    dx = x - cx
    dy = y - cy
    face_dist = np.sqrt(dx**2 + dy**2)

    # Calculate maximum distance (corner to face center)
    corners = np.array([
        [0, 0],
        [0, h-1],
        [w-1, 0],
        [w-1, h-1]
    ])
    corner_dists = np.sqrt(np.sum((corners - np.array([cx, cy]))**2, axis=1))
    max_dist = np.max(corner_dists)

    # Define face radius (protected area)
    face_radius = face_size / 2

    # Create stretch factor mask
    stretch_factor = np.zeros_like(face_dist)

    # Inside face area - no stretching
    face_mask = face_dist < face_radius

    # Outside face area - gradually increase stretching
    non_face_mask = ~face_mask
    relative_dist = np.zeros_like(face_dist)
    relative_dist[non_face_mask] = (face_dist[non_face_mask] - face_radius) / (max_dist - face_radius)

    # Apply stronger stretch effect
    stretch_factor[non_face_mask] = relative_dist[non_face_mask] * relative_dist[non_face_mask] * strength * 1.5

    # Calculate source coordinates
    src_x = x - dx * stretch_factor
    src_y = y - dy * stretch_factor

    # Ensure coordinates are within image bounds
    src_x = np.clip(src_x, 0, w - 1.001)
    src_y = np.clip(src_y, 0, h - 1.001)

    # Use OpenCV's remap function for faster interpolation
    map_x = src_x.astype(np.float32)
    map_y = src_y.astype(np.float32)

    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    return result

def apply_radial_blur(image, face_center, strength=15, face_protection=1.0):
    """Apply radial blur that keeps the face clear while blurring surroundings"""
    if strength <= 0:
        return image.copy()

    h, w = image.shape[:2]
    cx, cy, face_size = face_center[:3]

    # Create a mask for the face area
    y, x = np.mgrid[0:h, 0:w]
    dx = x - cx
    dy = y - cy
    dist = np.sqrt(dx**2 + dy**2)

    # Define clear radius (protected area)
    clear_radius = face_size / 2

    # Create face protection mask
    face_mask = np.ones((h, w), dtype=np.float32)

    # Inside face area - no blur
    face_area = dist < clear_radius
    face_mask[face_area] = 0

    # Outside face area - gradual blur based on distance
    non_face_area = ~face_area
    blur_factor = np.maximum(0, face_protection * (1.0 - (dist[non_face_area] - clear_radius) / (3 * clear_radius)))
    face_mask[non_face_area] = blur_factor

    # For faster processing, use fewer blur levels
    if USE_FAST_BLUR:
        blur_levels = max(1, min(3, strength // 2))
        step = max(1, strength // blur_levels)
    else:
        blur_levels = max(1, strength // 2)
        step = 2

    # Create a copy of the image
    result = image.copy()

    # Apply multiple levels of blur
    for i in range(1, blur_levels + 1):
        kernel_size = 2 * i * step + 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # Calculate weight for this blur level
        weight = face_mask * (i / blur_levels)
        inv_weight = 1.0 - weight

        # Apply weighted blur
        for c in range(3):
            result[:, :, c] = result[:, :, c] * inv_weight + blurred[:, :, c] * weight

    return result.astype(np.uint8)

# ===== TRANSITION CREATION =====

def process_frame(args):
    """Process a single frame for the transition - used with multiprocessing"""
    img, face_center, zoom_progress, effect_progress, is_first_image = args

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
    blur_strength = int(BLUR_STRENGTH * effect_progress)
    if blur_strength > 0:
        result = apply_radial_blur(
            result,
            scaled_face_center,
            strength=blur_strength,
            face_protection=BLUR_FACE_PROTECTION
        )

    return result

def create_zoom_transition(img1, img2, face_center1, face_center2):
    """Create a zoom transition between two images with fisheye effect.

    The transition zooms in toward the face in the first image.
    Then, it switches to the second image, which starts zoomed in on its face
    and then zooms out to reveal the full second image.

    Args:
        img1: First image
        img2: Second image
        face_center1: (x, y, size) of the face in the first image
        face_center2: (x, y, size) of the face in the second image

    Returns:
        List of frames for the transition
    """
    frames = []

    # Hold the first image
    first_frame = img1.copy()
    frames.extend([first_frame] * HOLD_FRAMES)

    # Calculate transition frames (excluding hold frames)
    transition_frames = TOTAL_FRAMES - 2 * HOLD_FRAMES

    # We'll use a single continuous transition instead of two halves
    # This creates a smoother effect from zooming in on img1 to zooming in on img2

    # Prepare arguments for parallel processing
    frame_args = []

    # Create transition frames
    for i in range(transition_frames):
        # Calculate progress (0 to 1)
        progress = i / (transition_frames - 1)

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

    # Process frames in parallel
    if NUM_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Process frames in parallel
            transition_frames = list(executor.map(process_frame, frame_args))
    else:
        # Process frames sequentially
        transition_frames = [process_frame(args) for args in frame_args]

    # Add transition frames to the result
    frames.extend(transition_frames)

    # Hold the second image
    last_frame = img2.copy()
    frames.extend([last_frame] * HOLD_FRAMES)

    return frames

def get_random_music_file():
    """Get a random music file from the music directory"""
    music_files = glob.glob(os.path.join(MUSIC_DIR, '*.mp3'))
    if not music_files:
        print_warning("No music files found in the music directory.")
        return None

    return random.choice(music_files)

# ===== BATCH PROCESSING =====

def process_batch(images):
    """Process a batch of images and create a video"""
    if len(images) < 2:
        print_error("Need at least 2 images to create a transition.")
        return None

    print_header("STARTING BATCH PROCESSING")
    print_info(f"Processing {len(images)} images")

    # Clear output directory for new run
    for f in glob.glob(os.path.join(OUTPUT_DIR, '*.jpg')):
        os.remove(f)
    print_info("Cleared previous output frames")

    # Create output directories for frames and comparisons
    os.makedirs(os.path.join(OUTPUT_DIR, 'frames'), exist_ok=True)
    if CREATE_COMPARISON:
        os.makedirs(os.path.join(OUTPUT_DIR, 'comparisons'), exist_ok=True)

    # Initialize face detection model once
    global face_detection_model
    face_detection_model = FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    # Detect faces in all images first
    face_centers = {}
    print_info("Detecting faces in all images...")
    for img_path in tqdm(images, desc=f"{Colors.BLUE}Detecting faces{Colors.ENDC}", unit="image",
                        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.GREEN, Colors.ENDC)):
        face_centers[img_path] = detect_face_cached(img_path)

    # Process each pair of consecutive images
    frame_count = 0
    all_transition_frames = []

    for i in tqdm(range(len(images)-1), desc=f"{Colors.BLUE}Processing image pairs{Colors.ENDC}", unit="pair",
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.GREEN, Colors.ENDC)):
        img_path1 = images[i]
        img_path2 = images[i+1]

        # Load consecutive images
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        # Skip if either image failed to load
        if img1 is None or img2 is None:
            print_error(f"Failed to load images: {img_path1} or {img_path2}")
            continue

        # Resize images to the same dimensions if needed
        if img1.shape[:2] != img2.shape[:2]:
            h, w = img1.shape[:2]
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Get pre-detected faces
        face_center1 = face_centers[img_path1]
        face_center2 = face_centers[img_path2]

        print_info(f"Generating {TOTAL_FRAMES} frames for transition from:")
        print(f"  {Colors.CYAN}➤ {os.path.basename(img_path1)}{Colors.ENDC} → {Colors.CYAN}{os.path.basename(img_path2)}{Colors.ENDC}")
        print(f"  {Colors.GREEN}• Face in image 1: {face_center1[:3]}{Colors.ENDC}")
        print(f"  {Colors.GREEN}• Face in image 2: {face_center2[:3]}{Colors.ENDC}")

        # Create zoom transition
        transition_frames = create_zoom_transition(img1, img2, face_center1, face_center2)

        # Save frames with progress bar
        frame_paths = []
        for frame_idx, frame in enumerate(tqdm(transition_frames,
                                              desc=f"{Colors.GREEN}Saving frames for transition {i+1}/{len(images)-1}{Colors.ENDC}",
                                              unit="frame",
                                              bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.GREEN, Colors.ENDC))):
            # Save frame with high quality
            output_path = os.path.join(OUTPUT_DIR, 'frames', f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frame_paths.append(output_path)
            frame_count += 1

            # Create comparison frame if enabled
            if CREATE_COMPARISON and frame_idx % COMPARISON_INTERVAL == 0:
                # Create a side-by-side comparison
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                h_comp = max(h1, h2)
                w_comp1 = int(w1 * (h_comp / h1))
                w_comp2 = int(w2 * (h_comp / h2))

                # Resize images for comparison
                img1_comp = cv2.resize(img1, (w_comp1, h_comp), interpolation=cv2.INTER_LANCZOS4)
                img2_comp = cv2.resize(img2, (w_comp2, h_comp), interpolation=cv2.INTER_LANCZOS4)

                # Resize current frame to match height
                frame_comp = cv2.resize(frame, (w_comp1, h_comp), interpolation=cv2.INTER_LANCZOS4)

                # Create comparison image
                comparison = np.zeros((h_comp, w_comp1*3, 3), dtype=np.uint8)
                comparison[:, :w_comp1] = img1_comp
                comparison[:, w_comp1:w_comp1*2] = frame_comp
                comparison[:, w_comp1*2:] = img2_comp

                # Add labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(comparison, "Source", (10, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(comparison, f"Frame {frame_idx}", (w_comp1 + 10, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(comparison, "Target", (w_comp1*2 + 10, 30), font, 1, (255, 255, 255), 2)

                # Save the comparison image in the comparisons subfolder
                comp_path = os.path.join(OUTPUT_DIR, 'comparisons', f'comparison_{frame_count-1:06d}.jpg')
                cv2.imwrite(comp_path, comparison)

        all_transition_frames.extend(frame_paths)

    # Create video using FFmpeg
    output_pattern = os.path.join(OUTPUT_DIR, 'frames', 'frame_%06d.jpg')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_video = os.path.join(OUTPUT_DIR, f'VID_{FPS}fps_{timestamp}.mp4')

    # Check if frames were generated
    if all_transition_frames:
        print_header("CREATING VIDEO")
        print_info(f"Creating video from {len(all_transition_frames)} frames...")

        # Create a progress bar for video creation
        with tqdm(total=1, desc=f"{Colors.BLUE}Creating video{Colors.ENDC}", unit="video",
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.GREEN, Colors.ENDC)) as video_pbar:
            # Get a random music file if enabled
            music_file = None
            if ADD_BACKGROUND_MUSIC:
                music_file = get_random_music_file()
                if music_file:
                    print_info(f"Adding background music: {os.path.basename(music_file)}")

            # Create FFmpeg command
            ffmpeg_cmd = f'ffmpeg -y -framerate {FPS} -i "{output_pattern}" '

            # Add audio input if music file is available
            if music_file:
                ffmpeg_cmd += f'-i "{music_file}" '

            # Add video settings with resolution
            width, height = VIDEO_RESOLUTION
            ffmpeg_cmd += f'-c:v libx264 -vf "scale={width}:{height},format=yuv420p,unsharp=3:3:1.0:3:3:0.5" '
            ffmpeg_cmd += f'-preset veryslow -tune film -crf {VIDEO_QUALITY} '

            # Add audio settings if music file is available
            if music_file:
                # Add audio codec and use shortest option to trim music to video length
                ffmpeg_cmd += f'-c:a aac -b:a 192k -shortest '

            # Output file
            ffmpeg_cmd += f'"{output_video}"'

            # Run FFmpeg command
            print_info(f"Running FFmpeg to create {FPS}fps video...")
            os.system(ffmpeg_cmd)
            video_pbar.update(1)
            print_success(f"Created video: {output_video}")

            print_header("CLEANUP")
            # Delete all temporary images
            print_info("Cleaning up temporary files...")

            # Delete frame images
            frames = glob.glob(os.path.join(OUTPUT_DIR, 'frames', '*.jpg'))
            for frame in tqdm(frames, desc=f"{Colors.BLUE}Deleting frame images{Colors.ENDC}", unit="image",
                             bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.GREEN, Colors.ENDC)):
                os.remove(frame)

            # Delete comparison images if they were created
            if CREATE_COMPARISON:
                comparisons = glob.glob(os.path.join(OUTPUT_DIR, 'comparisons', '*.jpg'))
                for comp in tqdm(comparisons, desc=f"{Colors.BLUE}Deleting comparison images{Colors.ENDC}", unit="image",
                                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.GREEN, Colors.ENDC)):
                    os.remove(comp)

            # Move processed images to done directory if specified
            if DONE_DIR:
                os.makedirs(DONE_DIR, exist_ok=True)
                print_info(f"Moving processed images to {DONE_DIR}...")
                for img_path in tqdm(images, desc=f"{Colors.BLUE}Moving processed images{Colors.ENDC}", unit="image",
                                    bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.GREEN, Colors.ENDC)):
                    # Get the base filename
                    base_name = os.path.basename(img_path)
                    # Move the file
                    shutil.move(img_path, os.path.join(DONE_DIR, base_name))
    else:
        print_error("No frames were generated. Check if input images were processed correctly.")
        return None

    return output_video

def process_images():
    """Main processing function with batch processing and continuous monitoring"""
    # Print configuration
    print_header("SLIDESHOW CREATOR")

    # Print configuration in a nice table format
    terminal_width = shutil.get_terminal_size().columns
    config_width = min(80, terminal_width - 4)

    print(f"{Colors.CYAN}{Colors.BOLD}{'CONFIGURATION'.center(config_width)}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'─' * config_width}{Colors.ENDC}")

    # Input/Output settings
    print(f"{Colors.CYAN}{'INPUT/OUTPUT SETTINGS'.ljust(30)}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Input directory:{Colors.ENDC} {Colors.YELLOW}{INPUT_DIR}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Output directory:{Colors.ENDC} {Colors.YELLOW}{OUTPUT_DIR}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Done directory:{Colors.ENDC} {Colors.YELLOW}{DONE_DIR}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Music directory:{Colors.ENDC} {Colors.YELLOW}{MUSIC_DIR}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'─' * config_width}{Colors.ENDC}")

    # Processing settings
    print(f"{Colors.CYAN}{'PROCESSING SETTINGS'.ljust(30)}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• CPU cores:{Colors.ENDC} {Colors.YELLOW}{NUM_WORKERS}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Min images per batch:{Colors.ENDC} {Colors.YELLOW}{MIN_IMAGES}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Wait time between checks:{Colors.ENDC} {Colors.YELLOW}{WAIT_TIME} seconds{Colors.ENDC}")
    print(f"{Colors.CYAN}{'─' * config_width}{Colors.ENDC}")

    # Transition settings
    print(f"{Colors.CYAN}{'TRANSITION SETTINGS'.ljust(30)}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• FPS:{Colors.ENDC} {Colors.YELLOW}{FPS}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Transition duration:{Colors.ENDC} {Colors.YELLOW}{TRANSITION_DURATION} seconds{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Hold duration:{Colors.ENDC} {Colors.YELLOW}{HOLD_DURATION} seconds{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Total frames per transition:{Colors.ENDC} {Colors.YELLOW}{TOTAL_FRAMES}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'─' * config_width}{Colors.ENDC}")

    # Effect settings
    print(f"{Colors.CYAN}{'EFFECT SETTINGS'.ljust(30)}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Zoom amount:{Colors.ENDC} {Colors.YELLOW}{ZOOM_AMOUNT}x{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Blur strength:{Colors.ENDC} {Colors.YELLOW}{BLUR_STRENGTH}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Stretch amount:{Colors.ENDC} {Colors.YELLOW}{STRETCH_AMOUNT}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'─' * config_width}{Colors.ENDC}")

    # Video settings
    print(f"{Colors.CYAN}{'VIDEO SETTINGS'.ljust(30)}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Video quality (CRF):{Colors.ENDC} {Colors.YELLOW}{VIDEO_QUALITY}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Resolution:{Colors.ENDC} {Colors.YELLOW}{VIDEO_RESOLUTION[0]}x{VIDEO_RESOLUTION[1]}{Colors.ENDC}")
    print(f"  {Colors.BLUE}• Background music:{Colors.ENDC} {Colors.YELLOW}{ADD_BACKGROUND_MUSIC}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'─' * config_width}{Colors.ENDC}")

    # Create necessary directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DONE_DIR, exist_ok=True)
    os.makedirs(MUSIC_DIR, exist_ok=True)

    # Check if input directory is empty
    if not glob.glob(os.path.join(INPUT_DIR, '*.*')):
        print_warning(f"Input directory '{INPUT_DIR}' is empty. Please add some images.")
        return

    # Continuous monitoring and batch processing
    batch_count = 0
    while True:
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
            images.extend(glob.glob(os.path.join(INPUT_DIR, ext.upper())))

        # Sort images by name (important for consistent batching)
        images.sort()

        if len(images) >= MIN_IMAGES:
            batch_count += 1
            print_header(f"BATCH #{batch_count}")
            
            # Select exactly MIN_IMAGES for this batch
            images_to_process = images[:MIN_IMAGES]
            print_info(f"Processing batch of {len(images_to_process)} images (out of {len(images)} available).")

            output_video = process_batch(images_to_process) # process_batch moves these images

            if output_video:
                print_success(f"Batch #{batch_count} complete! Video saved to: {os.path.basename(output_video)}")
            else:
                print_error(f"Batch #{batch_count} processing failed.")
            # After processing (success or fail), the loop continues to re-evaluate image availability.

        elif images: # Some images found, but less than MIN_IMAGES (and MIN_IMAGES >= 2)
            if len(images) < 2:
                print_info(f"Found {len(images)} image. Need at least 2 for a transition, and {MIN_IMAGES} to start a batch.")
            else: # 2 <= len(images) < MIN_IMAGES
                print_info(f"Found {len(images)} images. Waiting for at least {MIN_IMAGES} images to start a batch.")
            print_waiting(f"Checking again in {INPUT_DIR}", WAIT_TIME)
        
        else: # No images found
            print_warning(f"No images found in {INPUT_DIR}.")
            print_waiting(f"Waiting for images in {INPUT_DIR}", WAIT_TIME)

if __name__ == "__main__":
    try:
        process_images()
    except KeyboardInterrupt:
        print_info("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
