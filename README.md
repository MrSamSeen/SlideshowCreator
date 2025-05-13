# Slideshow Creator

![SamSeen Logo](https://img.shields.io/badge/Slideshow_Creator-by_SamSeen_Solutions-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.6%2B-blue?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=for-the-badge)

## Description

Slideshow Creator is a Python script that creates smooth zoom transitions between images with face detection. It detects faces in each image, creates transitions by zooming out from the first image and zooming in to the second image, applies special effects during transitions, and combines all frames into a high-quality video with optional background music.

## Main Features

- Detects faces in each image.
- Creates transitions by zooming out from the first image and zooming in to the second image.
- Applies special effects during transitions (fisheye/stretching and radial blur).
- Combines all frames into a high-quality video with optional background music.

## Installation and Running

### Dependencies

- Python 3
- OpenCV (`opencv-python-headless`)
- NumPy (`numpy`)
- MediaPipe (`mediapipe`)
- tqdm (`tqdm`)

### Installation Steps

1. Ensure Python 3 is installed.
2. Install required Python packages using pip:
   ```bash
   pip install opencv-python-headless numpy mediapipe tqdm
   ```

### Running the Script

1. Place images in the `input` directory.
2. Place background music in the `music` directory (optional).
3. Run the script:
   ```bash
   python slideshow_creator.py
   ```

## Dependencies or Prerequisites

- Python 3
- OpenCV (`opencv-python-headless`)
- NumPy (`numpy`)
- MediaPipe (`mediapipe`)
- tqdm (`tqdm`)
- multiprocessing (built-in)
- glob (built-in)
- shutil (built-in)
- datetime (built-in)
- math (built-in)
- random (built-in)
- time (built-in)

## How to Contribute

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Make your changes and commit them.
- Push your changes to your fork.
- Create a pull request to the main repository.

## Additional Information

- **Configuration Parameters:** The script uses configuration parameters defined in the script for input/output settings, processing settings, transition settings, effect parameters, video output parameters, and performance parameters.
- **Terminal Formatting:** The script includes a `Colors` class for terminal colors and functions for printing headers, success messages, info messages, warning messages, error messages, and waiting messages.
- **Face Detection:** Uses MediaPipe for face detection.
- **Special Effects:** Includes functions for applying fisheye/stretching and radial blur effects.
- **Transition Creation:** Includes functions for processing frames and creating zoom transitions between images.
- **Batch Processing:** Includes functions for processing batches of images and creating a video.
