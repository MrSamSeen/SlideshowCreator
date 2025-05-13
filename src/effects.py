import cv2
import numpy as np
from .config import USE_FAST_BLUR, STRETCH_FACE_PROTECTION

# ===== SPECIAL EFFECTS =====

def apply_fisheye_effect(image, face_center, strength=0.3):
    """Apply fisheye/stretching effect that stretches toward edges while protecting face area"""
    # If strength is too low, don't apply any effect
    if strength < 0.001:
        return image.copy()

    h, w = image.shape[:2]
    cx, cy, face_size = face_center[:3]

    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Calculate distance from face center for each pixel
    dx = x_coords - cx
    dy = y_coords - cy
    dist_from_face = np.sqrt(dx**2 + dy**2)

    # Calculate maximum distance (corner to face center)
    # This helps normalize the effect
    corners = np.array([
        [0, 0],
        [0, h-1],
        [w-1, 0],
        [w-1, h-1]
    ])
    corner_dists = np.sqrt(np.sum((corners - np.array([cx, cy]))**2, axis=1))
    max_dist = np.max(corner_dists)
    if max_dist == 0: max_dist = 1 # Avoid division by zero if image is 1x1 and face is at (0,0)

    # Normalized distance from face (0 at face center, 1 at furthest point from face)
    norm_dist_from_face = dist_from_face / max_dist

    # Calculate protection factor based on distance from face center
    # Protection is high near the face, low further away
    # This uses a sigmoid-like curve for smooth transition
    # The STRETCH_FACE_PROTECTION controls how large the protected area is
    # A higher value means a larger protected area around the face.
    protection_radius_factor = face_size * STRETCH_FACE_PROTECTION * 0.5 # Adjust as needed
    if protection_radius_factor < 1e-6: # Avoid division by zero or extreme values if face_size or STRETCH_FACE_PROTECTION is zero
        protection = 0.0 # No protection if radius is effectively zero
    else:
        protection = 1.0 - np.exp(-(dist_from_face**2) / (2 * (protection_radius_factor**2)))

    # Calculate displacement: pixels are pushed away from the center
    # The strength of displacement increases with distance from the center
    # and is modulated by the overall effect strength and protection factor
    displacement_strength = strength * norm_dist_from_face * protection

    # Avoid division by zero if dist_from_face is zero
    # Create masks for non-zero distances to prevent division by zero
    mask = dist_from_face != 0

    # Initialize new coordinates with original coordinates
    new_x = x_coords.copy()
    new_y = y_coords.copy()

    # Apply displacement only where dist_from_face is not zero
    # Simplified and potentially more stable calculation:
    # The original form (dx[mask] / dist_from_face[mask]) * (displacement_strength[mask] * dist_from_face[mask])
    # is equivalent to dx[mask] * displacement_strength[mask] when dist_from_face[mask] is not zero.
    new_x[mask] = x_coords[mask] - dx[mask] * displacement_strength[mask]
    new_y[mask] = y_coords[mask] - dy[mask] * displacement_strength[mask]

    # Ensure new coordinates are within bounds
    new_x = np.clip(new_x, 0, w - 1)
    new_y = np.clip(new_y, 0, h - 1)

    # Remap the image
    remapped_image = cv2.remap(image, new_x, new_y, cv2.INTER_LANCZOS4)

    return remapped_image

def apply_radial_blur(image, face_center, strength=10, face_protection=0.5):
    """Apply radial blur effect, stronger at edges, protecting the face area."""
    if strength <= 0:
        return image.copy()

    h, w = image.shape[:2]
    cx, cy, face_size = face_center[:3]

    # Create a mask for the face area to protect it from blur
    # The size of the protected area is determined by face_size and face_protection
    protection_radius = int(face_size * face_protection * 0.75) # 0.75 is an adjustment factor

    # Create a circular mask for the face
    face_mask = np.zeros((h, w), dtype=np.float32)
    if protection_radius > 0:
        center_coordinates = (cx, cy)
        cv2.circle(face_mask, center_coordinates, protection_radius, (0,0,0), cv2.LINE_AA)
        # Feather the mask edges for a smoother transition
        face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0) # Adjust kernel size as needed
    face_mask_expanded = face_mask[:, :, np.newaxis] # For broadcasting with 3-channel image

    # Generate multiple blurred versions of the image
    # Using a faster blur for intermediate steps if enabled
    kernel_base_size = 3  # Smallest kernel for blur

    # Create a base blurred image
    k_offset = strength // 2
    k_size = kernel_base_size + k_offset

    # Ensure kernel size is odd for GaussianBlur and positive
    if not USE_FAST_BLUR:
        k_size = k_size + 1 if k_size % 2 == 0 else k_size
    k_size = max(1, k_size)

    if USE_FAST_BLUR:
        # For boxFilter, ddepth=-1 maintains the image depth, kernel size must be a tuple
        blurred_img = cv2.boxFilter(image, -1, (k_size, k_size))
    else:
        # For GaussianBlur, sigmaX=0 lets OpenCV compute it based on kernel size
        blurred_img = cv2.GaussianBlur(image, (k_size, k_size), 0)

    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    # Calculate distance from the center for each pixel
    dx = x_coords - cx
    dy = y_coords - cy
    dist_from_center = np.sqrt(dx**2 + dy**2)

    # Normalize distance (0 at center, 1 at corners/edges)
    max_dist = np.sqrt(max(cx, w - cx)**2 + max(cy, h - cy)**2)
    if max_dist == 0: max_dist = 1 # Avoid division by zero
    norm_dist = dist_from_center / max_dist

    # Create a weight map for blending: more blur further from center
    # This map will be combined with the face protection mask
    blur_intensity_map = norm_dist
    # Expand dims for broadcasting with 3-channel image
    blur_intensity_map_expanded = blur_intensity_map[:, :, np.newaxis]

    # Combine blur intensity with face protection
    # Where face_mask is 1 (face area), blur_weight becomes low
    # Where face_mask is 0 (non-face area), blur_weight is determined by blur_intensity_map
    final_blur_weights = blur_intensity_map_expanded * np.subtract(1.0, face_mask_expanded)

    # Blend the original image and the heavily blurred image using the final_blur_weights
    # result = original * (1 - weight) + blurred * weight
    result = image.astype(np.float32) * (1.0 - final_blur_weights) + blurred_img.astype(np.float32) * final_blur_weights
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result