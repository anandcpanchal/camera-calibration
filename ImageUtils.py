import cv2
import numpy as np
import os

# Termination criteria for cornerSubPix and optimization algorithms
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def loadImages(folder_name_path):
    """
    Load all images from a specified folder.

    Args:
        folder_name_path (str): Path to the folder containing images.

    Returns:
        list of dict: A list of dictionaries, each containing 'path', 'name', and 'data' (image as numpy.ndarray).
                      Returns an empty list if folder doesn't exist or no images are loaded.
    """
    if not os.path.isdir(folder_name_path):
        print(f"Error: Folder '{folder_name_path}' not found.")
        return []

    files = os.listdir(folder_name_path)
    images_loaded_info = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    print(f"Loading images from '{folder_name_path}':")
    for f_name in sorted(files):  # Sort to ensure consistent order
        if f_name.lower().endswith(supported_extensions):
            image_path = os.path.join(folder_name_path, f_name)
            image_data = cv2.imread(image_path)
            if image_data is not None:
                images_loaded_info.append(dict(path=image_path, name= f_name, data= image_data))
                # print(f"  Loaded '{f_name}' (shape: {image_data.shape})") # Reduce console noise
            else:
                print(f"  Error loading image '{f_name}'")
        # else:
        # print(f"  Skipping non-image file or unsupported extension: '{f_name}'") # Reduce noise

    if not images_loaded_info:
        print(f"No images were loaded from '{folder_name_path}'. Check file paths and formats.")
    return images_loaded_info

def getImagesPoints(images_list, h_corners, w_corners):
    """
    Detect chessboard corners in a list of images.

    Args:
        images_list (list of numpy.ndarray): List of input images (BGR format).
        h_corners (int): Number of internal corners along the height of the chessboard.
        w_corners (int): Number of internal corners along the width of the chessboard.

    Returns:
        list of numpy.ndarray: A list where each element is an array of detected 2D image corners
                               (shape (N, 2)) for one calibration image where corners were found.
                               Returns an empty list if no corners are found in any image.
    """
    all_image_corners = []
    for i, image_data in enumerate(images_list):
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (w_corners, h_corners), None)

        if ret == True:
            # Refine corner locations
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            corners_subpix = corners_subpix.reshape(-1, 2)  # Reshape to (N, 2)
            all_image_corners.append(corners_subpix)
            # print(f"Chessboard corners found and refined for image {i+1}.") # Moved to main for better context
        # else:
        # print(f"Chessboard corners not found for image {i+1}.") # Moved to main
    return all_image_corners

def displayCorners(images_list_display, all_image_corners_display, h_corners, w_corners, save_folder_path):
    """
    Display and save images with detected chessboard corners drawn.
    Assumes images_list_display and all_image_corners_display are corresponding lists
    (i.e., images_list_display[i] is the image for all_image_corners_display[i]).

    Args:
        images_list_display (list of numpy.ndarray): List of original images for which corners were found.
        all_image_corners_display (list of numpy.ndarray): List of detected corners for each image in images_list_display.
        h_corners (int): Number of internal corners along the height.
        w_corners (int): Number of internal corners along the width.
        save_folder_path (str): Folder path to save the images with drawn corners.
    """
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
        print(f"Created directory: {save_folder_path}")

    if len(images_list_display) != len(all_image_corners_display):
        print("Warning in displayCorners: Mismatch between number of images and corner sets. Skipping display.")
        return

    for i, corners_for_image in enumerate(all_image_corners_display):
        image_to_draw_on = images_list_display[i].copy()  # Use the image corresponding to this set of corners
        corners_float32 = np.float32(corners_for_image.reshape(-1, 1, 2))
        cv2.drawChessboardCorners(image_to_draw_on, (w_corners, h_corners), corners_float32, True)

        display_width = 800
        if image_to_draw_on.shape[1] > display_width:
            scale_factor = display_width / image_to_draw_on.shape[1]
            img_resized = cv2.resize(image_to_draw_on, (display_width, int(image_to_draw_on.shape[0] * scale_factor)))
        else:
            img_resized = image_to_draw_on

        filename = os.path.join(save_folder_path, f"corners_img_{i:03d}.png")
        cv2.imwrite(filename, img_resized)
        # print(f"Saved image with corners to {filename}") # Reduce console noise, implied by main loop

    # cv2.destroyAllWindows() # If using imshow

