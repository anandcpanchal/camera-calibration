import scipy
import scipy.optimize
from MathHelper import *
from ImageUtils import *
import logging

# --- Debug Logging Configuration ---
DEBUG_LOGGING_ENABLED = True  # Set to False to disable debug logging to file
DEEP_DEBUG_LOGGING_ENABLED = False  # Set to False to disable deep debug logging (recursive functions) to file
DEBUG_LOG_FILENAME = "calibration_debug.log"

def setup_debug_logger(log_file_path, enabled=True):
    """Sets up a dedicated logger for debug information."""
    logger = logging.getLogger("calibration_logger")
    # Prevent multiple handlers if function is called multiple times (though not in this script's flow)
    if logger.hasHandlers():
        logger.handlers.clear()

    if enabled:
        logger.setLevel(logging.DEBUG)
        # File Handler for debug log file
        try:
            fh = logging.FileHandler(log_file_path, mode='w') # mode='w' to overwrite log each run
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.info("Debug logger initialized. Logging to file: %s", log_file_path)
        except Exception as e:
            # Fallback to console if file handler fails for some reason
            print(f"Error setting up file logger at {log_file_path}: {e}. Debug logs might not be saved to file.")
            logger.setLevel(logging.CRITICAL + 1) # Disable if file setup failed
            logger.addHandler(logging.NullHandler())

    else:
        logger.setLevel(logging.CRITICAL + 1) # Effectively disable it by setting level too high
        logger.addHandler(logging.NullHandler()) # Avoid 'No handlers' warning
    return logger

def extractParamsFromA_kc(A_matrix_val, kc_coeffs_val, logger=None):
    """
    Extract individual parameters from the intrinsic matrix A_matrix_val and distortion coefficients kc_coeffs_val.
    """
    if logger: logger.debug(f"Entering extractParamsFromA_kc.\nA_matrix_val:\n{A_matrix_val}\nkc_coeffs_val: {kc_coeffs_val}")
    alpha_val = A_matrix_val[0, 0]
    gamma_val = A_matrix_val[0, 1]
    beta_val = A_matrix_val[1, 1]
    u0_val = A_matrix_val[0, 2]
    v0_val = A_matrix_val[1, 2]

    k1_val = kc_coeffs_val[0] if len(kc_coeffs_val) > 0 else 0.0
    k2_val = kc_coeffs_val[1] if len(kc_coeffs_val) > 1 else 0.0

    x0_params = np.array([alpha_val, gamma_val, beta_val, u0_val, v0_val, k1_val, k2_val], dtype=np.float64)
    if logger: logger.debug(f"Extracted parameters x0: {x0_params}")
    if logger: logger.debug("Exiting extractParamsFromA_kc.")
    return x0_params

def retrieveA_kc_fromParams(x_params_vec, logger=None):
    """
    Reconstruct the intrinsic matrix A_matrix_val and distortion coefficients kc_coeffs_val from a parameter vector.
    """
    if logger: logger.debug(f"Entering retrieveA_kc_fromParams with x_params_vec: {x_params_vec}")
    alpha_p, gamma_p, beta_p, u0_p, v0_p, k1_p, k2_p = x_params_vec
    A_matrix_val = np.array([
        [alpha_p, gamma_p, u0_p],
        [0.0,     beta_p,  v0_p],
        [0.0,     0.0,   1.0]
    ], dtype=np.float64)
    kc_coeffs_val = np.array([k1_p, k2_p], dtype=np.float64)
    if logger: logger.debug(f"Retrieved A_matrix_val:\n{A_matrix_val}\nRetrieved kc_coeffs_val: {kc_coeffs_val}")
    if logger: logger.debug("Exiting retrieveA_kc_fromParams.")
    return A_matrix_val, kc_coeffs_val


def projectPointsWithDistortion(world_points_3d_homo_coords, RT_extrinsic_mat, A_intrinsic_mat, kc_coeffs_arr, logger=None):
    """
    Projects 3D world points to 2D image points considering lens distortion.
    """
    # Extensive logging here might be too verbose for every call, but entry/exit is good.
    # if logger: logger.debug("Entering projectPointsWithDistortion.")

    wp_3d_h = world_points_3d_homo_coords
    if wp_3d_h.ndim == 1:
        wp_3d_h = wp_3d_h.reshape(4,1)
    elif wp_3d_h.shape[0] != 4 and wp_3d_h.shape[1] == 4:
        wp_3d_h = wp_3d_h.T

    camera_coords_homo = np.dot(RT_extrinsic_mat, wp_3d_h)

    Zc = camera_coords_homo[2, :]
    epsilon = 1e-9
    valid_Z_mask = Zc > epsilon

    x_normalized = np.full_like(Zc, np.nan)
    y_normalized = np.full_like(Zc, np.nan)

    if np.any(valid_Z_mask):
        x_normalized[valid_Z_mask] = camera_coords_homo[0, valid_Z_mask] / Zc[valid_Z_mask]
        y_normalized[valid_Z_mask] = camera_coords_homo[1, valid_Z_mask] / Zc[valid_Z_mask]

    k1 = kc_coeffs_arr[0] if len(kc_coeffs_arr) > 0 else 0.0
    k2 = kc_coeffs_arr[1] if len(kc_coeffs_arr) > 1 else 0.0

    r_sq = x_normalized**2 + y_normalized**2
    distortion_factor = (1.0 + k1 * r_sq + k2 * r_sq**2)

    x_distorted_norm = x_normalized * distortion_factor
    y_distorted_norm = y_normalized * distortion_factor

    alpha_f = A_intrinsic_mat[0,0]
    gamma_s = A_intrinsic_mat[0,1]
    u0_c    = A_intrinsic_mat[0,2]
    beta_f  = A_intrinsic_mat[1,1]
    v0_c    = A_intrinsic_mat[1,2]

    u_pixels = alpha_f * x_distorted_norm + gamma_s * y_distorted_norm + u0_c
    v_pixels = beta_f * y_distorted_norm + v0_c

    projected_points_final = np.vstack((u_pixels, v_pixels)).T
    # if logger: logger.debug("Exiting projectPointsWithDistortion.")
    return projected_points_final


def reprojectPointsAndGetError(A_intrinsic_mat, kc_coeffs_arr, all_RT_extrinsics_list,
                               all_observed_image_corners_list, world_corners_2d_coords, logger=None):
    """
    Reproject world points to image plane and calculate reprojection error.
    """
    if logger: logger.debug(f"Entering reprojectPointsAndGetError. Number of views: {len(all_RT_extrinsics_list)}")
    total_squared_error_sum = 0.0
    total_points_count = 0
    all_reprojected_points_distorted_list_final = []

    world_corners_3d_homo = np.hstack((
        world_corners_2d_coords,
        np.zeros((world_corners_2d_coords.shape[0], 1)),
        np.ones((world_corners_2d_coords.shape[0], 1))
    )).T
    if logger: logger.debug(f"World corners 3D Homogeneous (shape {world_corners_3d_homo.shape}):\nCols:\n{world_corners_3d_homo}")

    for i, observed_corners_img in enumerate(all_observed_image_corners_list):
        if logger: logger.debug(f"  Processing view {i+1}/{len(all_observed_image_corners_list)} for reprojection error.")
        if i >= len(all_RT_extrinsics_list) or all_RT_extrinsics_list[i] is None:
            if logger: logger.warning(f"  Skipping view {i} in error calculation: missing or invalid extrinsics.")
            all_reprojected_points_distorted_list_final.append(np.full_like(observed_corners_img, np.nan))
            continue
        RT_extrinsic_mat = all_RT_extrinsics_list[i]
        if logger: logger.debug(f"    Using RT for view {i}:\n{RT_extrinsic_mat}")

        reprojected_points_img = projectPointsWithDistortion(world_corners_3d_homo, RT_extrinsic_mat, A_intrinsic_mat, kc_coeffs_arr, logger=None) # Inner loop, less logger
        all_reprojected_points_distorted_list_final.append(reprojected_points_img)
        if logger: logger.debug(f"    Reprojected points for view {i} (shape {reprojected_points_img.shape}). Reprojected Points:\n{reprojected_points_img}")


        valid_mask = ~np.isnan(reprojected_points_img).any(axis=1)
        current_reprojected_pts = reprojected_points_img[valid_mask]
        current_observed_pts = observed_corners_img[valid_mask] # Ensure observed points also filtered if reproj had NaNs

        if current_observed_pts.shape[0] == 0:
            if logger: logger.warning(f"    No valid points to compare for view {i} after NaN filtering in reprojection.")
            continue

        errors_per_point_sq = np.sum((current_observed_pts - current_reprojected_pts)**2, axis=1)
        image_error_sum_sq = np.sum(errors_per_point_sq)
        if logger: logger.debug(f"    View {i}: Sum of squared errors = {image_error_sum_sq:.4f} for {current_observed_pts.shape[0]} points.")

        total_squared_error_sum += image_error_sum_sq
        total_points_count += current_observed_pts.shape[0]

    if total_points_count == 0:
        if logger: logger.error("No points available to calculate total reprojection error.")
        return np.inf, all_reprojected_points_distorted_list_final

    mean_squared_error_val = total_squared_error_sum / total_points_count
    root_mean_squared_error_val = np.sqrt(mean_squared_error_val)
    if logger: logger.debug(f"Total points for error: {total_points_count}, Total sum of squared_errors: {total_squared_error_sum:.4f}")
    if logger: logger.debug(f"Exiting reprojectPointsAndGetError. Final RMSE: {root_mean_squared_error_val:.4f}")

    return root_mean_squared_error_val, all_reprojected_points_distorted_list_final


def lossFunctionOptimization(params_x0_vec, initial_all_RT_extrinsics_list, all_observed_image_corners_list,
                             world_corners_2d_coords, fixed_extrinsics_flag=False, logger=None):
    """
    Loss function for non-linear optimization (scipy.optimize.least_squares).
    """
    # This function is called many times by the optimizer, so logging inside can be very verbose.
    # Log entry/exit or key parameter changes if needed, but be cautious.
    if logger and DEEP_DEBUG_LOGGING_ENABLED: logger.debug(f"lossFunctionOptimization called. fixed_extrinsics_flag: {fixed_extrinsics_flag}, params_x0_vec[:7]: {params_x0_vec[:7]}")

    num_intrinsic_params = 7
    A_intrinsic_mat, kc_coeffs_arr = retrieveA_kc_fromParams(params_x0_vec[:num_intrinsic_params], logger=None) # No logger for frequent calls

    current_all_RT_extrinsics = []
    if fixed_extrinsics_flag:
        current_all_RT_extrinsics = initial_all_RT_extrinsics_list
    else:
        num_extrinsic_params_per_image = 6
        extr_params_flat = params_x0_vec[num_intrinsic_params:]
        num_images = len(all_observed_image_corners_list)

        expected_extr_len = num_images * num_extrinsic_params_per_image
        if len(extr_params_flat) != expected_extr_len:
            if logger: logger.error(f"CRITICAL in lossFunction: Mismatch in extrinsic parameters length. Expected {expected_extr_len}, got {len(extr_params_flat)}. This will likely fail optimization.")
            # This is a critical error for the optimizer. Return large residuals.
            total_expected_residuals = sum(obs.shape[0] * 2 for obs in all_observed_image_corners_list)
            return np.full(total_expected_residuals, 1e9) # Return a correctly shaped array of large errors


        for i in range(num_images):
            start_idx = i * num_extrinsic_params_per_image
            end_idx_r = start_idx + 3
            end_idx_t = start_idx + 6

            rodrigues_vec = extr_params_flat[start_idx : end_idx_r]
            translation_vec = extr_params_flat[end_idx_r : end_idx_t]

            try:
                R_matrix_val, _ = cv2.Rodrigues(rodrigues_vec)
            except cv2.error as e: # Catch potential errors from Rodrigues conversion (e.g. bad input vector)
                if logger: logger.error(f"cv2.Rodrigues failed for image {i} during optimization: {e}. Using identity R, zero t.")
                R_matrix_val = np.eye(3) # Fallback R
                # translation_vec might also be problematic, but use as is.
            RT_matrix_val = np.hstack((R_matrix_val, translation_vec.reshape(3,1)))
            current_all_RT_extrinsics.append(RT_matrix_val)

    all_residuals_list = []
    world_corners_3d_homo = np.hstack((
        world_corners_2d_coords,
        np.zeros((world_corners_2d_coords.shape[0], 1)),
        np.ones((world_corners_2d_coords.shape[0], 1))
    )).T

    total_points_processed_for_residuals = 0

    for i, observed_corners_img in enumerate(all_observed_image_corners_list):
        num_points_this_image = observed_corners_img.shape[0]
        total_points_processed_for_residuals += num_points_this_image

        if i >= len(current_all_RT_extrinsics) or current_all_RT_extrinsics[i] is None:
            if logger and i < 5: # Log only for first few occurrences to avoid flooding
                logger.warning(f"LossFunc: Missing/invalid extrinsics for image {i}. Appending large error residuals ({num_points_this_image*2} values).")
            all_residuals_list.extend(np.full(num_points_this_image * 2, 1e6)) # Large error for these points
            continue

        RT_extrinsic_mat = current_all_RT_extrinsics[i]
        reprojected_points_img = projectPointsWithDistortion(world_corners_3d_homo, RT_extrinsic_mat, A_intrinsic_mat, kc_coeffs_arr, logger=None)

        # Handle NaNs from projection consistently for fixed-size residual vector
        nan_mask = np.isnan(reprojected_points_img)
        # Create residuals array, then fill problem spots
        residuals_img_pairs = observed_corners_img - reprojected_points_img

        # For any point where reprojection was NaN, set residual to a large value
        # nan_points_mask is (N,2), True where NaN. .any(axis=1) makes it (N,), True if either x or y is NaN.
        nan_points_indices = np.where(nan_mask.any(axis=1))[0]
        if len(nan_points_indices) > 0:
            if logger and i < 5 and np.random.rand() < 0.1 : # Sporadic logging for NaNs during optimization
                 logger.debug(f"LossFunc view {i}: {len(nan_points_indices)} NaN projections. Assigning large residuals.")
            for pt_idx in nan_points_indices:
                residuals_img_pairs[pt_idx, :] = 1e6 # Large error for both x and y components of this point

        all_residuals_list.extend(residuals_img_pairs.ravel())

    final_residuals = np.array(all_residuals_list)
    if logger and DEEP_DEBUG_LOGGING_ENABLED: logger.debug(f"lossFunctionOptimization returning {final_residuals.shape[0]} residuals. Examples: {final_residuals}")
    return final_residuals


def calibrate(input_path, output_path, corner_width_count, corner_height_count, square_side=20):

    # --- Logger Setup ---
    # Ensure output_path exists before setting up logger if log file is inside it
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        # No logger yet, so print
        print(f"Created base results directory: {output_path}")

    debug_file_name = os.path.join(output_path, DEBUG_LOG_FILENAME)
    logger = setup_debug_logger(debug_file_name, DEBUG_LOGGING_ENABLED)
    logger.info("--- Starting Zhang Camera Calibration Script ---")
    logger.info(f"Image source folder: {input_path}")
    logger.info(f"Results save folder: {output_path}")

    # --- Chessboard Configuration ---
    h_corners = corner_height_count
    w_corners = corner_width_count
    logger.info(f"Chessboard inner corners: height={h_corners}, width={w_corners}. Square side: {square_side} units.")

    if not os.path.exists(input_path):
        logger.critical(f"Image folder '{input_path}' not found. Please update the path. Exiting.")
        print(f"ERROR: Image folder '{input_path}' not found. Please update the path.")
        return

    # 1. Load Images
    logger.info("--- Stage 1: Loading Images ---")
    loaded_images_info = loadImages(input_path, logger=logger) # List of dicts: {'path', 'name', 'data'}
    if not loaded_images_info:
        logger.critical("No images loaded. Exiting.")
        return
    logger.info(f"Successfully loaded {len(loaded_images_info)} image files.")

    # 2. Find Chessboard Corners
    logger.info("--- Stage 2: Finding Chessboard Corners ---")
    all_image_points_raw, successful_indices = getImagesPoints(loaded_images_info, h_corners, w_corners, logger=logger)

    if not all_image_points_raw:
        logger.critical("No chessboard corners found in any image. Calibration cannot proceed. Exiting.")
        return

    # Filter original image list and names to match successful corner detections
    images_for_calibration = [loaded_images_info[i]['data'] for i in successful_indices]
    original_filenames_for_calibration = [loaded_images_info[i]['name'] for i in successful_indices]
    logger.info(f"Found corners in {len(all_image_points_raw)} out of {len(loaded_images_info)} images. Proceeding with these.")
    logger.debug(f"Original filenames of images with detected corners: {original_filenames_for_calibration}")


    # 3. Get World Coordinates
    logger.info("--- Stage 3: Generating World Coordinates ---")
    world_points_2d = getWorldPoints(square_side, h_corners, w_corners)
    logger.debug(f"Generated world points (Z=0), shape: {world_points_2d.shape}. Worldf points:\n{world_points_2d}")

    # 4. Display Detected Corners (Optional Visual Check)
    logger.info("--- Stage 4: Saving Images with Detected Corners ---")
    displayCorners_save_folder = os.path.join(output_path, "1_detected_corners")
    displayCorners(images_for_calibration, all_image_points_raw, h_corners, w_corners, displayCorners_save_folder, original_filenames_for_calibration, logger=logger)
    logger.info(f"Detected corner images saved in '{displayCorners_save_folder}'")

    # --- Initial Parameter Estimation (Linear Method) ---
    logger.info("--- Stage 5: Initial Parameter Estimation (Linear Method) ---")

    # 5a. Compute Homographies
    logger.info("Step 5a: Calculating initial homographies (H)...")
    all_H_computed = getAllH(all_image_points_raw, square_side, h_corners, w_corners, logger=logger)

    # Filter out views where H computation failed
    all_H_valid = []
    all_image_points_for_H_valid = []
    images_for_H_valid = []
    original_filenames_for_H_valid = []

    for i, H_val in enumerate(all_H_computed):
        if H_val is not None:
            all_H_valid.append(H_val)
            all_image_points_for_H_valid.append(all_image_points_raw[i])
            images_for_H_valid.append(images_for_calibration[i])
            original_filenames_for_H_valid.append(original_filenames_for_calibration[i])
        else:
            logger.warning(f"Invalid homography for image '{original_filenames_for_calibration[i]}' (index {i}). Skipping this view for B matrix calculation.")

    if len(all_H_valid) < 2:
        logger.critical(f"Not enough valid homographies ({len(all_H_valid)}) to proceed (need at least 2). Calibration aborted.")
        return
    logger.info(f"Proceeding with {len(all_H_valid)} views with valid homographies.")
    # Update lists to only include valid views from this point onwards
    all_H_initial = all_H_valid
    all_image_points = all_image_points_for_H_valid
    images_for_calibration_current = images_for_H_valid
    original_filenames_current = original_filenames_for_H_valid


    # 5b. Estimate B matrix
    logger.info("Step 5b: Calculating initial B matrix...")
    B_initial = getB_matrix_from_V(all_H_initial, logger=logger)
    if B_initial is None:
        logger.critical("Failed to estimate B matrix. Calibration aborted.")
        return
    logger.info(f"Estimated initial B matrix:\n{B_initial}")

    # 5c. Estimate Intrinsic Matrix A
    logger.info("Step 5c: Calculating initial intrinsic matrix (A)...")
    A_initial = getA_intrinsic_matrix(B_initial, logger=logger)
    if A_initial is None:
        logger.critical("Failed to compute initial intrinsic matrix A. Calibration aborted.")
        return
    logger.info(f"Initial estimated A (intrinsic matrix):\n{A_initial}")

    # 5d. Estimate Extrinsic Parameters (R, t)
    logger.info("Step 5d: Calculating initial rotation (R) and translation (t) for each view...")
    all_RT_computed = getRotationAndTrans(A_initial, all_H_initial, logger=logger)

    # Filter views where RT computation failed
    all_RT_valid = []
    all_image_points_for_RT_valid = []
    images_for_RT_valid = []
    original_filenames_for_RT_valid = []
    all_H_for_RT_valid = [] # Keep H consistent for optimization step if needed

    for i, RT_val in enumerate(all_RT_computed):
        if RT_val is not None:
            all_RT_valid.append(RT_val)
            all_image_points_for_RT_valid.append(all_image_points[i]) # all_image_points was already filtered for H
            images_for_RT_valid.append(images_for_calibration_current[i])
            original_filenames_for_RT_valid.append(original_filenames_current[i])
            all_H_for_RT_valid.append(all_H_initial[i])
        else:
            logger.warning(f"Failed to compute extrinsics for view {i} (image '{original_filenames_current[i]}'). Skipping this view.")

    if len(all_RT_valid) < 1:
        logger.critical(f"Not enough valid extrinsics ({len(all_RT_valid)}) to proceed. Calibration aborted.")
        return
    logger.info(f"Proceeding with {len(all_RT_valid)} views with valid extrinsics for initial error calculation.")
    # Update lists again
    all_RT_initial = all_RT_valid
    all_image_points = all_image_points_for_RT_valid # This is now the set of points for views with valid H and RT
    images_for_calibration_final_set = images_for_RT_valid
    original_filenames_final_set = original_filenames_for_RT_valid
    all_H_final_set = all_H_for_RT_valid


    # 5e. Initial Distortion Coefficients
    kc_initial = np.array([0.0, 0.0], dtype=np.float64)
    logger.info(f"Initial distortion coefficients (kc_initial): {kc_initial}")

    # 5f. Calculate Initial Reprojection Error
    logger.info("Step 5f: Calculating initial reprojection error...")
    initial_reprojection_error, _ = reprojectPointsAndGetError(
        A_initial, kc_initial, all_RT_initial,
        all_image_points, world_points_2d, logger=logger
    )
    logger.info(f"Initial Root Mean Squared Reprojection Error: {initial_reprojection_error:.4f} pixels (based on {len(all_RT_initial)} views)")
    print(f"Initial Root Mean Squared Reprojection Error: {initial_reprojection_error:.4f} pixels")


    # --- Non-linear Optimization (Refinement) ---
    logger.info("--- Stage 6: Non-linear Optimization (Refinement) ---")
    x0_for_optimization = extractParamsFromA_kc(A_initial, kc_initial, logger=logger)
    logger.debug(f"Parameters for optimization (x0): {x0_for_optimization}")

    # Extrinsics are fixed during this optimization based on all_RT_initial
    optimization_args = (all_RT_initial, all_image_points, world_points_2d, True, logger) # True for fixed_extrinsics, pass logger
    logger.info("Optimizing intrinsic parameters and distortion coefficients (extrinsics fixed to initial estimates)...")

    A_optimized, kc_optimized = A_initial, kc_initial # Default to initial if optimization fails severely
    all_RT_optimized = all_RT_initial # Default

    try:
        logger.debug("Starting scipy.optimize.least_squares call...")
        optimization_result = scipy.optimize.least_squares(
            fun=lossFunctionOptimization, # Will use its own logger instance if passed
            x0=x0_for_optimization,
            method="trf", # alternate methods : lm / dogbox
            args=optimization_args, # Pass tuple of args
            verbose=0 # Set to 0 to reduce console spam, rely on our logger. 1 or 2 for scipy's own verbosity.
        )
        logger.debug(f"scipy.optimize.least_squares call finished. Cost: {optimization_result.cost}, Status: {optimization_result.status}, Message: {optimization_result.message}")
    except Exception as e:
        logger.error(f"Optimization call crashed with exception: {e}", exc_info=True)
        print(f"Error during optimization: {e}")
        print("Skipping optimization. Using initial parameters as final.")
        logger.info("Optimization skipped due to runtime error. Using initial parameters for A, kc, and RT.")
        # A_optimized, kc_optimized, all_RT_optimized already defaulted
    else:
        logger.debug("Optimization try-block completed without raising an exception.")
        if not optimization_result.success:
            logger.warning(f"Optimization reported as not successful. Message: {optimization_result.message}. Status: {optimization_result.status}")
            # print("Optimization failed or did not converge. Using initial parameters for A and kc.") # Already logged
            # print("Message:", optimization_result.message) # Already logged
            logger.info("Using initial A and kc due to optimization non-convergence. RTs remain initial.")
            # A_optimized, kc_optimized already defaulted to initial
        else:
            logger.info(f"Optimization reported as successful. Final cost: {optimization_result.cost:.4e}")
            # print("Optimization successful.") # Already logged
            optimized_params = optimization_result.x
            logger.debug(f"Optimized raw parameters (x0 form): {optimized_params}")
            A_optimized, kc_optimized = retrieveA_kc_fromParams(optimized_params, logger=logger)
            logger.info(f"Optimized A_matrix:\n{A_optimized}")
            logger.info(f"Optimized kc_coeffs: {kc_optimized}")

        logger.info("Re-calculating extrinsics with new A_optimized (if optimization changed A)...")
        # Use all_H_final_set which corresponds to the views that had valid initial RTs
        all_RT_optimized_temp = getRotationAndTrans(A_optimized, all_H_final_set, logger=logger)

        # Filter again for RTs that might have failed with the new A_optimized
        final_all_RT_optimized = []
        final_all_image_points_for_opt_RT = []
        final_images_for_opt_RT = []
        final_original_filenames_for_opt_RT = []

        for i, RT_val in enumerate(all_RT_optimized_temp):
            if RT_val is not None:
                final_all_RT_optimized.append(RT_val)
                final_all_image_points_for_opt_RT.append(all_image_points[i]) # all_image_points is already filtered for initial RTs
                final_images_for_opt_RT.append(images_for_calibration_final_set[i])
                final_original_filenames_for_opt_RT.append(original_filenames_final_set[i])
            else:
                logger.warning(f"Failed to compute extrinsics with A_optimized for view originally '{original_filenames_final_set[i]}'. Skipping this view for final error/visualization.")

        all_RT_optimized = final_all_RT_optimized
        # Update the image points and related lists to match the final set of valid RTs
        all_image_points_final_error_calc = final_all_image_points_for_opt_RT
        images_for_visualization = final_images_for_opt_RT
        original_filenames_for_visualization = final_original_filenames_for_opt_RT

        if not all_RT_optimized:
            logger.warning("No valid extrinsics could be computed with A_optimized. Final error calculation might be problematic.")
            # Fallback to initial RTs if all fail, though this is unlikely if A_optimized is reasonable.
            # For now, proceed with empty all_RT_optimized if that's the case.
            all_image_points_final_error_calc = [] # No points if no RTs
            images_for_visualization = []
            original_filenames_for_visualization = []


    logger.info(f"Final A_optimized to be used:\n{A_optimized}")
    logger.info(f"Final kc_optimized to be used: {kc_optimized}")
    logger.info(f"Number of views for final error calculation / visualization: {len(all_RT_optimized)}")


    # 11. Calculate Final Reprojection Error
    logger.info("--- Stage 7: Calculating Final Reprojection Error ---")
    if not all_RT_optimized or not all_image_points_final_error_calc:
        logger.warning("No valid views (RTs or image points) for final error calculation. Skipping.")
        final_reprojection_error = np.inf
        all_reprojected_points_final = []
    else:
        final_reprojection_error, all_reprojected_points_final = reprojectPointsAndGetError(
            A_optimized, kc_optimized, all_RT_optimized,
            all_image_points_final_error_calc, world_points_2d, logger=logger
        )
    logger.info(f"Final Root Mean Squared Reprojection Error: {final_reprojection_error:.4f} pixels (based on {len(all_RT_optimized)} views)")
    print(f"Final Root Mean Squared Reprojection Error: {final_reprojection_error:.4f} pixels")

    print(f"\nComparison of Reprojection Error:")
    print(f"  Initial RMSE: {initial_reprojection_error:.4f} pixels (based on {len(all_RT_initial)} views)") # all_RT_initial was the set used for this calc
    print(f"  Final RMSE  : {final_reprojection_error:.4f} pixels (based on {len(all_RT_optimized)} views)")
    logger.info(f"Initial RMSE: {initial_reprojection_error:.4f} (views: {len(all_RT_initial)}), Final RMSE: {final_reprojection_error:.4f} (views: {len(all_RT_optimized)})")


    # 12. Undistort images and draw reprojected points
    logger.info("--- Stage 8: Undistorting Images and Drawing Reprojected Points ---")
    K_final_cv = np.array(A_optimized, dtype=np.float32)
    D_final_cv = np.array([kc_optimized[0], kc_optimized[1], 0, 0, 0], dtype=np.float32)
    logger.debug(f"Using K for undistortion:\n{K_final_cv}\nUsing D for undistortion: {D_final_cv}")

    undistorted_save_folder = os.path.join(output_path, "2_undistorted_reprojected")
    if not os.path.exists(undistorted_save_folder):
        os.makedirs(undistorted_save_folder)
        logger.info(f"Created directory for undistorted images: {undistorted_save_folder}")

    if not images_for_visualization:
        logger.warning("No images available for final visualization (likely due to prior errors in RT calculation).")
    else:
        logger.info(f"Processing {len(images_for_visualization)} images for undistortion and point drawing.")

    for i in range(len(images_for_visualization)):
        image_to_process = images_for_visualization[i].copy()
        current_original_filename = original_filenames_for_visualization[i]
        observed_pts_for_this_img = all_image_points_final_error_calc[i] # these are the original detected corners

        # Reprojected points for this image (from the final error calculation)
        reprojected_pts_for_this_img = None
        if all_reprojected_points_final and i < len(all_reprojected_points_final):
            reprojected_pts_for_this_img = all_reprojected_points_final[i]

        logger.debug(f"  Visualizing for image '{current_original_filename}' (index {i}).")

        h_img, w_img = image_to_process.shape[:2]
        try:
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K_final_cv, D_final_cv, (w_img, h_img), alpha=0.5, newImgSize=(w_img,h_img))
            logger.debug(f"    Optimal new camera matrix for '{current_original_filename}':\n{new_camera_matrix}\n    ROI: {roi}")
            img_undistorted = cv2.undistort(image_to_process, K_final_cv, D_final_cv, None, new_camera_matrix)
        except cv2.error as e:
            logger.error(f"    cv2 error during undistortion/getOptimalNewCameraMatrix for '{current_original_filename}': {e}. Skipping visualization for this image.")
            continue

        # Draw reprojected points (model based, transformed to undistorted image plane)
        if reprojected_pts_for_this_img is not None and not np.isnan(reprojected_pts_for_this_img).all():
            points_to_draw_distorted_model = np.float32(reprojected_pts_for_this_img).reshape(-1,1,2)
            valid_pts_mask_model = ~np.isnan(points_to_draw_distorted_model).any(axis=(1,2))
            if np.any(valid_pts_mask_model) :
                try:
                    undistorted_reproj_pts = cv2.undistortPoints(points_to_draw_distorted_model[valid_pts_mask_model], K_final_cv, D_final_cv, P=new_camera_matrix)
                    for pt_coords_undistorted in undistorted_reproj_pts:
                        x_val, y_val = int(round(pt_coords_undistorted[0][0])), int(round(pt_coords_undistorted[0][1]))
                        cv2.circle(img_undistorted, (x_val, y_val), radius=4, color=(0, 0, 255), thickness=-1) # Red for reprojected
                except cv2.error as e:
                     logger.warning(f"    cv2 error undistorting reprojected points for '{current_original_filename}': {e}")


        # Draw observed points (original detected, transformed to undistorted image plane)
        observed_corners_reshaped = np.float32(observed_pts_for_this_img).reshape(-1,1,2)
        try:
            undistorted_observed_pts = cv2.undistortPoints(observed_corners_reshaped, K_final_cv, D_final_cv, P=new_camera_matrix)
            for obs_pt_coords_undistorted in undistorted_observed_pts:
                x_obs, y_obs = int(round(obs_pt_coords_undistorted[0][0])), int(round(obs_pt_coords_undistorted[0][1]))
                cv2.circle(img_undistorted, (x_obs, y_obs), radius=6, color=(0, 255, 0), thickness=1) # Green for observed
        except cv2.error as e:
            logger.warning(f"    cv2 error undistorting observed points for '{current_original_filename}': {e}")


        base_name_viz, ext_viz = os.path.splitext(current_original_filename)
        save_filename_viz = os.path.join(undistorted_save_folder, f"{base_name_viz}_undistorted_reproj{ext_viz}")
        try:
            cv2.imwrite(save_filename_viz, img_undistorted)
            logger.debug(f"    Saved undistorted image with points for '{current_original_filename}' to '{save_filename_viz}'")
        except Exception as e:
            logger.error(f"    Failed to save visualized image '{save_filename_viz}': {e}")


    logger.info(f"Undistorted images with points potentially saved in '{undistorted_save_folder}' (check logs for errors).")
    logger.info("--- Zhang Camera Calibration Script Finished ---")
    print(f"Calibration process complete. Check detailed logs L {debug_file_name}")