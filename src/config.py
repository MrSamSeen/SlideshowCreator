# ===== CONFIGURATION PARAMETERS =====
import multiprocessing

# Input/Output Settings
INPUT_DIR = "input"           # Directory containing input images
OUTPUT_DIR = "output"         # Directory for output files
DONE_DIR = "done"             # Directory to move processed images
MUSIC_DIR = "music"           # Directory containing background music files

# Processing Settings
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Use all but one CPU core
MIN_IMAGES = 15               # Minimum number of images required to start a batch
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