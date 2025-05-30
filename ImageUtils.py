import cv2
import numpy as np
import os

# Termination criteria for cornerSubPix and optimization algorithms
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def loadImages(folder_name_path, logger=None):
    """
    Load all images from a specified folder.
    """
    if logger: logger.debug(f"Entering loadImages. Folder: '{folder_name_path}'")
    if not os.path.isdir(folder_name_path):
        if logger: logger.error(f"Folder '{folder_name_path}' not found.")
        print(f"Error: Folder '{folder_name_path}' not found.")
        return []

    files = os.listdir(folder_name_path)
    images_loaded_info = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    if logger: logger.debug(f"Found {len(files)} files/dirs in folder. Supported extensions: {supported_extensions}")

    for f_name in sorted(files):
        if f_name.lower().endswith(supported_extensions):
            image_path = os.path.join(folder_name_path, f_name)
            if logger: logger.debug(f"  Attempting to load image: '{image_path}'")
            image_data = cv2.imread(image_path)
            if image_data is not None:
                images_loaded_info.append({'path': image_path, 'name': f_name, 'data': image_data})
                if logger: logger.debug(f"    Successfully loaded '{f_name}' (shape: {image_data.shape})")
            else:
                if logger: logger.warning(f"    Error loading image '{f_name}' (cv2.imread returned None)")
        # else:
            # if logger: logger.debug(f"  Skipping non-image file or unsupported extension: '{f_name}'")

    if not images_loaded_info:
        if logger: logger.warning(f"No images were loaded from '{folder_name_path}'.")
        print(f"No images were loaded from '{folder_name_path}'. Check file paths and formats.")
    if logger: logger.debug(f"Exiting loadImages. Loaded {len(images_loaded_info)} images.")
    return images_loaded_info

def getImagesPoints(images_list, h_corners, w_corners, logger=None):
    """
    Detect chessboard corners in a list of images.
    """
    if logger: logger.debug(f"Entering getImagesPoints. Number of images: {len(images_list)}, h_corners: {h_corners}, w_corners: {w_corners}")
    all_image_corners_list = [] # Stores corner arrays for images where detection was successful
    successful_image_indices = [] # Stores original indices of successfully processed images

    for i, image_data_dict in enumerate(images_list): # Assuming images_list is list of dicts from loadImages
        image_data = image_data_dict['data']
        image_name = image_data_dict['name']
        if logger: logger.debug(f"Processing image {i+1}/{len(images_list)} ('{image_name}') for corners.")

        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (w_corners, h_corners), None)

        if ret == True:
            if logger: logger.debug(f"  Initial corners found for '{image_name}'. Refining...")
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            corners_subpix_reshaped = corners_subpix.reshape(-1, 2)
            all_image_corners_list.append(corners_subpix_reshaped)
            successful_image_indices.append(i) # Store original index
            if logger: logger.debug(f"  Refined corners for '{image_name}'. Shape: {corners_subpix_reshaped.shape}. Added to list.")
        else:
            if logger: logger.info(f"  Chessboard corners NOT found for image '{image_name}'.")

    if logger: logger.debug(f"Exiting getImagesPoints. Found corners in {len(all_image_corners_list)} images.")
    return all_image_corners_list, successful_image_indices

def displayCorners(images_list_display, all_image_corners_display, h_corners, w_corners, save_folder_path, original_filenames=None, logger=None):
    """
    Display and save images with detected chessboard corners drawn.
    """
    if logger: logger.debug(f"Entering displayCorners. Saving to: {save_folder_path}")
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
        if logger: logger.info(f"Created directory for displaying corners: {save_folder_path}")

    if len(images_list_display) != len(all_image_corners_display):
        if logger: logger.error("Mismatch between number of images and corner sets in displayCorners. Skipping display.")
        return

    for i, corners_for_image in enumerate(all_image_corners_display):
        image_to_draw_on = images_list_display[i].copy()
        corners_float32 = np.float32(corners_for_image.reshape(-1, 1, 2))
        cv2.drawChessboardCorners(image_to_draw_on, (w_corners, h_corners), corners_float32, True)

        display_width = 800 # Target width for saved image
        if image_to_draw_on.shape[1] > display_width:
            scale_factor = display_width / image_to_draw_on.shape[1]
            img_resized = cv2.resize(image_to_draw_on, (display_width, int(image_to_draw_on.shape[0] * scale_factor)))
        else:
            img_resized = image_to_draw_on

        if original_filenames and i < len(original_filenames):
            base, ext = os.path.splitext(original_filenames[i])
            filename = os.path.join(save_folder_path, f"{base}_corners{ext}")
        else:
            filename = os.path.join(save_folder_path, f"corners_img_{i:03d}.png")

        cv2.imwrite(filename, img_resized)
        if logger: logger.debug(f"Saved image with corners to {filename}")
    if logger: logger.debug("Exiting displayCorners.")

