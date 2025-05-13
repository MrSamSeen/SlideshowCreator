import os
import cv2
import glob
import datetime
import random
import shutil
from tqdm import tqdm
import mediapipe as mp
from mediapipe.python.solutions.face_detection import FaceDetection

from .config import (
    OUTPUT_DIR, CREATE_COMPARISON, COMPARISON_INTERVAL, JPEG_QUALITY, 
    FPS, ADD_BACKGROUND_MUSIC, VIDEO_RESOLUTION, VIDEO_QUALITY, DONE_DIR, MUSIC_DIR
)
from .utils import (
    print_header, print_info, print_success, print_error, print_warning, Colors
)
from .face_detection import detect_face_cached
from .transitions import create_zoom_transition

face_detection_model_local = None # Local to this module to avoid global conflicts if imported elsewhere

def get_random_music_file():
    """Get a random music file from the music directory"""
    music_files = glob.glob(os.path.join(MUSIC_DIR, '*.mp3'))
    music_files.extend(glob.glob(os.path.join(MUSIC_DIR, '*.MP3')))
    music_files.extend(glob.glob(os.path.join(MUSIC_DIR, '*.wav')))
    music_files.extend(glob.glob(os.path.join(MUSIC_DIR, '*.WAV')))
    music_files.extend(glob.glob(os.path.join(MUSIC_DIR, '*.aac')))
    music_files.extend(glob.glob(os.path.join(MUSIC_DIR, '*.AAC')))
    if not music_files:
        print_warning(f"No music files found in {MUSIC_DIR} directory.")
        return None
    return random.choice(music_files)

def create_video_ffmpeg(image_pattern, output_video_path, music_file):
    """Creates a video from image frames using FFmpeg."""
    print_info(f"Creating video from frames matching: {image_pattern}")
    
    ffmpeg_cmd = f'ffmpeg -y -framerate {FPS} -i "{image_pattern}" '

    if music_file and ADD_BACKGROUND_MUSIC:
        print_info(f"Adding background music: {os.path.basename(music_file)}")
        ffmpeg_cmd += f'-i "{music_file}" '

    width, height = VIDEO_RESOLUTION
    ffmpeg_cmd += f'-c:v libx264 -vf "scale={width}:{height},format=yuv420p,unsharp=3:3:1.0:3:3:0.5" '
    ffmpeg_cmd += f'-preset veryslow -tune film -crf {VIDEO_QUALITY} '

    if music_file and ADD_BACKGROUND_MUSIC:
        ffmpeg_cmd += f'-c:a aac -b:a 192k -shortest '
    
    ffmpeg_cmd += f'"{output_video_path}"'

    print_info(f"Running FFmpeg command:")
    # print(ffmpeg_cmd) # Optional: print the command for debugging
    
    # Use a progress bar for the FFmpeg process if possible, or just execute
    # For simplicity, direct execution here. TQDM for os.system is tricky.
    os.system(ffmpeg_cmd)
    
    if os.path.exists(output_video_path):
        print_success(f"Video created: {output_video_path}")
        return True
    else:
        print_error(f"Failed to create video: {output_video_path}")
        return False

def process_batch(images_to_process):
    """Process a batch of images and create a video"""
    global face_detection_model_local

    if len(images_to_process) < 2:
        print_error("Need at least 2 images to create a transition.")
        return None

    print_info(f"Processing {len(images_to_process)} images for this batch.")

    # Ensure output directories exist
    frames_dir = os.path.join(OUTPUT_DIR, 'frames')
    comparisons_dir = os.path.join(OUTPUT_DIR, 'comparisons')
    os.makedirs(frames_dir, exist_ok=True)
    if CREATE_COMPARISON:
        os.makedirs(comparisons_dir, exist_ok=True)
    
    # Clear previous frames from this specific batch run if any (optional, depends on desired behavior)
    # For now, let's assume frames_dir is for the current batch and can be cleared or managed.
    # If multiple batches run concurrently or output_dir is shared, this needs more robust handling.
    # For simplicity, we'll let frames accumulate with unique names if not cleared by a higher-level process.

    # Initialize face detection model if not already done
    if face_detection_model_local is None:
        face_detection_model_local = FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

    # Detect faces
    face_centers = {}
    print_info("Detecting faces for the batch...")
    for img_path in tqdm(images_to_process, desc=f"{Colors.BLUE}Detecting faces{Colors.ENDC}", unit="image",
                        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.GREEN, Colors.ENDC)):
        # Pass the model to the cached function if it expects it, or ensure it uses its own global/local
        # detect_face_cached is designed to initialize its own model if needed.
        face_info = detect_face_cached(img_path)
        if face_info is None:
            print_error(f"Could not get face info for {img_path}. Skipping this image in pairs.")
            # Decide how to handle: skip image, use default, etc.
            # For now, it will lead to None in face_centers, handled later.
        face_centers[img_path] = face_info

    all_frame_paths = []
    frame_counter = 0 # Unique frame counter for this batch

    for i in tqdm(range(len(images_to_process) - 1), desc=f"{Colors.BLUE}Processing image pairs{Colors.ENDC}", unit="pair",
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.GREEN, Colors.ENDC)):
        img_path1 = images_to_process[i]
        img_path2 = images_to_process[i+1]

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        face_center1 = face_centers.get(img_path1)
        face_center2 = face_centers.get(img_path2)

        if img1 is None or img2 is None or face_center1 is None or face_center2 is None:
            print_error(f"Skipping pair due to missing image or face data: {os.path.basename(img_path1)}, {os.path.basename(img_path2)}")
            continue
        
        # Resize images to target video resolution
        target_h, target_w = VIDEO_RESOLUTION[1], VIDEO_RESOLUTION[0] # Use target_h, target_w for clarity
        
        # img1 and img2 are overwritten with resized versions
        img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Scale face coordinates from original image dimensions to target video dimensions
        # face_center1 and face_center2 are tuples from detect_face_cached: (cx_orig, cy_orig, size_orig, (orig_h, orig_w, _))
        
        # Unpack original face data for image 1
        cx1_orig, cy1_orig, size1_orig, (orig_h1, orig_w1, _) = face_center1 # face_center1 is the full tuple
        
        # Calculate scaling factors for image 1
        scale_w1 = float(target_w) / orig_w1 if orig_w1 > 0 else 1.0 # Ensure float division
        scale_h1 = float(target_h) / orig_h1 if orig_h1 > 0 else 1.0
        
        # Scale face center 1
        scaled_cx1 = int(cx1_orig * scale_w1)
        scaled_cy1 = int(cy1_orig * scale_h1)
        # Scale size: average of width and height scales applied to original size. Ensure size is at least 1.
        scaled_size1 = max(1, int(size1_orig * (scale_w1 + scale_h1) / 2.0)) 
        face_center1_scaled_for_transition = (scaled_cx1, scaled_cy1, scaled_size1) # This is the (cx, cy, size) tuple

        print_info(f"Img1 ({os.path.basename(img_path1)}): Orig ({cx1_orig},{cy1_orig},s{size1_orig}) on ({orig_w1}x{orig_h1}) -> Scaled ({scaled_cx1},{scaled_cy1},s{scaled_size1}) for ({target_w}x{target_h})")

        # Unpack original face data for image 2
        cx2_orig, cy2_orig, size2_orig, (orig_h2, orig_w2, _) = face_center2 # face_center2 is the full tuple

        # Calculate scaling factors for image 2
        scale_w2 = float(target_w) / orig_w2 if orig_w2 > 0 else 1.0 # Ensure float division
        scale_h2 = float(target_h) / orig_h2 if orig_h2 > 0 else 1.0

        # Scale face center 2
        scaled_cx2 = int(cx2_orig * scale_w2)
        scaled_cy2 = int(cy2_orig * scale_h2)
        # Scale size: average of width and height scales applied to original size. Ensure size is at least 1.
        scaled_size2 = max(1, int(size2_orig * (scale_w2 + scale_h2) / 2.0))
        face_center2_scaled_for_transition = (scaled_cx2, scaled_cy2, scaled_size2) # This is the (cx, cy, size) tuple
        
        print_info(f"Img2 ({os.path.basename(img_path2)}): Orig ({cx2_orig},{cy2_orig},s{size2_orig}) on ({orig_w2}x{orig_h2}) -> Scaled ({scaled_cx2},{scaled_cy2},s{scaled_size2}) for ({target_w}x{target_h})")

        # Use RESIZED images and SCALED face centers for the transition
        transition_frames = create_zoom_transition(img1, img2, face_center1_scaled_for_transition, face_center2_scaled_for_transition)

        for frame_idx, frame_data in enumerate(transition_frames):
            frame_path = os.path.join(frames_dir, f'frame_{frame_counter:06d}.jpg')
            cv2.imwrite(frame_path, frame_data, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            all_frame_paths.append(frame_path)
            
            if CREATE_COMPARISON and frame_idx % COMPARISON_INTERVAL == 0:
                # Create comparison frame (simplified, assumes img1, img2, frame_data are compatible for hstack)
                # This part needs careful resizing to a common height for proper side-by-side view.
                # For brevity, skipping detailed comparison frame generation here, assuming it's complex.
                pass # Placeholder for comparison frame logic
            frame_counter += 1

    if not all_frame_paths:
        print_error("No frames were generated for the video.")
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_video_filename = f'slideshow_{timestamp}.mp4'
    output_video_path = os.path.join(OUTPUT_DIR, output_video_filename)
    image_pattern_for_ffmpeg = os.path.join(frames_dir, 'frame_%06d.jpg')

    music_file_to_use = None
    if ADD_BACKGROUND_MUSIC:
        music_file_to_use = get_random_music_file()

    video_created = create_video_ffmpeg(image_pattern_for_ffmpeg, output_video_path, music_file_to_use)

    if video_created:
        print_info("Cleaning up temporary frame files...")
        for frame_file in tqdm(all_frame_paths, desc=f"{Colors.BLUE}Deleting frames{Colors.ENDC}", unit="frame"):
            try:
                os.remove(frame_file)
            except OSError as e:
                print_warning(f"Could not delete frame {frame_file}: {e}")
        
        # Optionally remove comparison frames dir or files if they were created
        # if CREATE_COMPARISON and os.path.exists(comparisons_dir):
        #     shutil.rmtree(comparisons_dir) # Example: remove whole dir

        if DONE_DIR:
            os.makedirs(DONE_DIR, exist_ok=True)
            print_info(f"Moving processed images to: {DONE_DIR}")
            for img_path in tqdm(images_to_process, desc=f"{Colors.BLUE}Moving images{Colors.ENDC}", unit="image"):
                try:
                    shutil.move(img_path, os.path.join(DONE_DIR, os.path.basename(img_path)))
                except Exception as e:
                    print_warning(f"Could not move {img_path} to {DONE_DIR}: {e}")
        return output_video_path
    else:
        print_error("Video creation failed. Frames might still be in frames_dir.")
        return None