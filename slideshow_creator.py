#!/usr/bin/env python3
"""
Slideshow Creator - Main script

This script serves as the main entry point for the slideshow creation application.
It initializes the process, monitors input images, and orchestrates batch processing.
"""

import os
import glob
import time
import shutil
import traceback

# Import from local src modules
from src import config
from src import utils
from src.video_processing import process_batch


def print_configuration():
    """Prints the current configuration settings to the terminal."""
    utils.print_header("SLIDESHOW CREATOR")

    terminal_width = shutil.get_terminal_size().columns
    config_width = min(80, terminal_width - 4)

    print(f"{utils.Colors.CYAN}{utils.Colors.BOLD}{'CONFIGURATION'.center(config_width)}{utils.Colors.ENDC}")
    print(f"{utils.Colors.CYAN}{'─' * config_width}{utils.Colors.ENDC}")

    # Input/Output settings
    print(f"{utils.Colors.CYAN}{'INPUT/OUTPUT SETTINGS'.ljust(30)}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Input directory:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.INPUT_DIR}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Output directory:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.OUTPUT_DIR}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Done directory:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.DONE_DIR}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Music directory:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.MUSIC_DIR}{utils.Colors.ENDC}")
    print(f"{utils.Colors.CYAN}{'─' * config_width}{utils.Colors.ENDC}")

    # Processing settings
    print(f"{utils.Colors.CYAN}{'PROCESSING SETTINGS'.ljust(30)}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• CPU cores:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.NUM_WORKERS}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Min images per batch:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.MIN_IMAGES}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Wait time between checks:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.WAIT_TIME} seconds{utils.Colors.ENDC}")
    print(f"{utils.Colors.CYAN}{'─' * config_width}{utils.Colors.ENDC}")

    # Transition settings
    print(f"{utils.Colors.CYAN}{'TRANSITION SETTINGS'.ljust(30)}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• FPS:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.FPS}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Transition duration:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.TRANSITION_DURATION} seconds{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Hold duration:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.HOLD_DURATION} seconds{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Total frames per transition:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.TOTAL_FRAMES}{utils.Colors.ENDC}")
    print(f"{utils.Colors.CYAN}{'─' * config_width}{utils.Colors.ENDC}")

    # Effect settings
    print(f"{utils.Colors.CYAN}{'EFFECT SETTINGS'.ljust(30)}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Zoom amount:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.ZOOM_AMOUNT}x{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Blur strength:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.BLUR_STRENGTH}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Stretch amount:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.STRETCH_AMOUNT}{utils.Colors.ENDC}")
    print(f"{utils.Colors.CYAN}{'─' * config_width}{utils.Colors.ENDC}")

    # Video settings
    print(f"{utils.Colors.CYAN}{'VIDEO SETTINGS'.ljust(30)}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Video quality (CRF):{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.VIDEO_QUALITY}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Resolution:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.VIDEO_RESOLUTION[0]}x{config.VIDEO_RESOLUTION[1]}{utils.Colors.ENDC}")
    print(f"  {utils.Colors.BLUE}• Background music:{utils.Colors.ENDC} {utils.Colors.YELLOW}{config.ADD_BACKGROUND_MUSIC}{utils.Colors.ENDC}")
    print(f"{utils.Colors.CYAN}{'─' * config_width}{utils.Colors.ENDC}")


def main_process_loop():
    """Main processing function with batch processing and continuous monitoring."""
    print_configuration()

    # Create necessary directories
    os.makedirs(config.INPUT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.DONE_DIR, exist_ok=True)
    os.makedirs(config.MUSIC_DIR, exist_ok=True)

    if not glob.glob(os.path.join(config.INPUT_DIR, '*.*')):
        utils.print_warning(f"Input directory '{config.INPUT_DIR}' is empty. Please add some images.")

    batch_count = 0
    while True:
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff',
                            '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(config.INPUT_DIR, ext)))

        images.sort()

        if len(images) >= config.MIN_IMAGES:
            batch_count += 1
            utils.print_header(f"BATCH #{batch_count}")
            
            images_for_this_batch = images[:config.MIN_IMAGES]
            utils.print_info(f"Processing batch of {len(images_for_this_batch)} images (out of {len(images)} available).")

            output_video_path = process_batch(images_for_this_batch)

            if output_video_path:
                utils.print_success(f"Batch #{batch_count} complete! Video saved to: {os.path.basename(output_video_path)}")
            else:
                utils.print_error(f"Batch #{batch_count} processing failed or no video produced.")
            
            # Small delay to allow file system operations to complete if necessary, 
            # and to prevent extremely rapid batch processing if new images arrive instantly.
            time.sleep(5) 
            
        elif images:
            if len(images) < 2:
                 utils.print_info(f"Found {len(images)} image. Need at least 2 for a transition, and {config.MIN_IMAGES} to start a batch.")
            else:
                utils.print_info(f"Found {len(images)} images. Waiting for at least {config.MIN_IMAGES} images to start a batch.")
            utils.print_waiting(f"Checking {config.INPUT_DIR} for more images", config.WAIT_TIME)
        
        else: # No images found
            utils.print_warning(f"No images found in {config.INPUT_DIR}.")
            utils.print_waiting(f"Waiting for images in {config.INPUT_DIR}", config.WAIT_TIME)

if __name__ == "__main__":
    try:
        main_process_loop()
    except KeyboardInterrupt:
        utils.print_info("\nProcess interrupted by user. Exiting.")
    except Exception as e:
        utils.print_error(f"An unexpected error occurred: {e}")
        traceback.print_exc()
