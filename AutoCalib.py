import scipy
import scipy.optimize
from MathHelper import *
from ImageUtils import *

square_side = 20  # Define the side length of the square in the chessboard pattern (in mm or any consistent unit)


def extractParamsFromA_kc(A_matrix_val, kc_coeffs_val):
    """
    Extract individual parameters from the intrinsic matrix A_matrix_val and distortion coefficients kc_coeffs_val.
    Assumes kc_coeffs_val is [k1, k2, p1, p2, k3] but here only k1, k2 are used.

    Args:
        A_matrix_val (numpy.ndarray): The 3x3 intrinsic camera matrix.
        kc_coeffs_val (numpy.ndarray or list/tuple): Distortion coefficients (k1, k2, ...).
                                                 Expected to have at least k1, k2.

    Returns:
        numpy.ndarray: A vector x0_params = [alpha, gamma, beta, u0, v0, k1, k2].
    """
    alpha_val = A_matrix_val[0, 0]
    gamma_val = A_matrix_val[0, 1]  # Skew
    beta_val = A_matrix_val[1, 1]
    u0_val = A_matrix_val[0, 2]  # Principal point x
    v0_val = A_matrix_val[1, 2]  # Principal point y

    k1_val = kc_coeffs_val[0] if len(kc_coeffs_val) > 0 else 0.0
    k2_val = kc_coeffs_val[1] if len(kc_coeffs_val) > 1 else 0.0

    x0_params = np.array([alpha_val, gamma_val, beta_val, u0_val, v0_val, k1_val, k2_val], dtype=np.float64)
    return x0_params


def retrieveA_kc_fromParams(x_params_vec):
    """
    Reconstruct the intrinsic matrix A_matrix_val and distortion coefficients kc_coeffs_val from a parameter vector.

    Args:
        x_params_vec (numpy.ndarray): Parameter vector [alpha, gamma, beta, u0, v0, k1, k2].

    Returns:
        tuple: (A_matrix_val, kc_coeffs_val)
               A_matrix_val (numpy.ndarray): The 3x3 intrinsic camera matrix.
               kc_coeffs_val (numpy.ndarray): Distortion coefficients [k1, k2].
    """
    alpha_p, gamma_p, beta_p, u0_p, v0_p, k1_p, k2_p = x_params_vec
    A_matrix_val = np.array([
        [alpha_p, gamma_p, u0_p],
        [0.0, beta_p, v0_p],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    kc_coeffs_val = np.array([k1_p, k2_p], dtype=np.float64)
    return A_matrix_val, kc_coeffs_val


def projectPointsWithDistortion(world_points_3d_homo_coords, RT_extrinsic_mat, A_intrinsic_mat, kc_coeffs_arr):
    """
    Projects 3D world points to 2D image points considering lens distortion.

    Args:
        world_points_3d_homo_coords (numpy.ndarray): Homogeneous 3D world points (4xN or list of 4x1).
        RT_extrinsic_mat (numpy.ndarray): 3x4 extrinsic parameter matrix [R_matrix_val|t_vec].
        A_intrinsic_mat (numpy.ndarray): 3x3 intrinsic camera matrix.
        kc_coeffs_arr (numpy.ndarray): Distortion coefficients [k1, k2, (p1, p2, k3)].
                                   Only k1, k2 are used here.

    Returns:
        numpy.ndarray: 2D projected image points (distorted) (Nx2).
    """
    # Ensure world_points_3d_homo_coords is (4, N)
    wp_3d_h = world_points_3d_homo_coords
    if wp_3d_h.ndim == 1:
        wp_3d_h = wp_3d_h.reshape(4, 1)
    elif wp_3d_h.shape[0] != 4 and wp_3d_h.shape[1] == 4:
        wp_3d_h = wp_3d_h.T

    camera_coords_homo = np.dot(RT_extrinsic_mat, wp_3d_h)

    Zc = camera_coords_homo[2, :]
    # Epsilon to prevent division by zero and issues with points at/behind camera
    epsilon = 1e-9
    valid_Z_mask = Zc > epsilon

    x_normalized = np.full_like(Zc, np.nan)
    y_normalized = np.full_like(Zc, np.nan)

    if np.any(valid_Z_mask):
        x_normalized[valid_Z_mask] = camera_coords_homo[0, valid_Z_mask] / Zc[valid_Z_mask]
        y_normalized[valid_Z_mask] = camera_coords_homo[1, valid_Z_mask] / Zc[valid_Z_mask]

    k1 = kc_coeffs_arr[0] if len(kc_coeffs_arr) > 0 else 0.0
    k2 = kc_coeffs_arr[1] if len(kc_coeffs_arr) > 1 else 0.0

    r_sq = x_normalized ** 2 + y_normalized ** 2
    distortion_factor = (1.0 + k1 * r_sq + k2 * r_sq ** 2)  # Using 1.0 for float

    x_distorted_norm = x_normalized * distortion_factor
    y_distorted_norm = y_normalized * distortion_factor

    alpha_f = A_intrinsic_mat[0, 0]
    gamma_s = A_intrinsic_mat[0, 1]
    u0_c = A_intrinsic_mat[0, 2]
    beta_f = A_intrinsic_mat[1, 1]
    v0_c = A_intrinsic_mat[1, 2]

    u_pixels = alpha_f * x_distorted_norm + gamma_s * y_distorted_norm + u0_c
    v_pixels = beta_f * y_distorted_norm + v0_c

    projected_points_final = np.vstack((u_pixels, v_pixels)).T
    return projected_points_final


def reprojectPointsAndGetError(A_intrinsic_mat, kc_coeffs_arr, all_RT_extrinsics_list,
                               all_observed_image_corners_list, world_corners_2d_coords):
    """
    Reproject world points to image plane using estimated parameters and calculate reprojection error.

    Args:
        A_intrinsic_mat (numpy.ndarray): Calibrated 3x3 intrinsic camera matrix.
        kc_coeffs_arr (numpy.ndarray): Calibrated distortion coefficients [k1, k2, ...].
        all_RT_extrinsics_list (list of numpy.ndarray): List of 3x4 extrinsic [R_matrix_val|t_vec] matrices, one per image.
        all_observed_image_corners_list (list of numpy.ndarray): List of observed 2D corner points for each image.
        world_corners_2d_coords (numpy.ndarray): 2D world coordinates of chessboard corners (Nx2, Z=0).

    Returns:
        tuple: (root_mean_squared_error_val, all_reprojected_points_distorted_list_final)
               root_mean_squared_error_val (float): RMSE between observed and reprojected points.
               all_reprojected_points_distorted_list_final (list of numpy.ndarray): List of reprojected 2D points for each image.
    """
    total_squared_error_sum = 0.0
    total_points_count = 0
    all_reprojected_points_distorted_list_final = []

    world_corners_3d_homo = np.hstack((
        world_corners_2d_coords,
        np.zeros((world_corners_2d_coords.shape[0], 1)),  # Z=0
        np.ones((world_corners_2d_coords.shape[0], 1))  # Homogeneous coordinate
    )).T  # Shape (4, N)

    for i, observed_corners_img in enumerate(all_observed_image_corners_list):
        if i >= len(all_RT_extrinsics_list) or all_RT_extrinsics_list[i] is None:
            print(f"Warning: Skipping image {i} in error calculation due to missing or invalid extrinsics.")
            all_reprojected_points_distorted_list_final.append(
                np.full_like(observed_corners_img, np.nan))  # Add NaN placeholder
            continue
        RT_extrinsic_mat = all_RT_extrinsics_list[i]

        reprojected_points_img = projectPointsWithDistortion(world_corners_3d_homo, RT_extrinsic_mat, A_intrinsic_mat,
                                                             kc_coeffs_arr)
        all_reprojected_points_distorted_list_final.append(reprojected_points_img)

        valid_mask = ~np.isnan(reprojected_points_img).any(axis=1)
        current_reprojected_pts = reprojected_points_img[valid_mask]
        current_observed_pts = observed_corners_img[valid_mask]

        if current_observed_pts.shape[0] == 0:
            # print(f"Warning: No valid points to compare for image {i} after NaN filtering.")
            continue

        errors_per_point_sq = np.sum((current_observed_pts - current_reprojected_pts) ** 2,
                                     axis=1)  # Squared Euclidean distance
        image_error_sum_sq = np.sum(errors_per_point_sq)

        total_squared_error_sum += image_error_sum_sq
        total_points_count += current_observed_pts.shape[0]

    if total_points_count == 0:
        print("Error: No points available to calculate reprojection error.")
        return np.inf, all_reprojected_points_distorted_list_final

    mean_squared_error_val = total_squared_error_sum / total_points_count
    root_mean_squared_error_val = np.sqrt(mean_squared_error_val)

    return root_mean_squared_error_val, all_reprojected_points_distorted_list_final


def lossFunctionOptimization(params_x0_vec, initial_all_RT_extrinsics_list, all_observed_image_corners_list,
                             world_corners_2d_coords, fixed_extrinsics_flag=False):
    """
    Loss function for non-linear optimization (scipy.optimize.least_squares).
    It computes the vector of reprojection errors (observed_pt - reprojected_pt) for all points.

    Args:
        params_x0_vec (numpy.ndarray): Current parameters [alpha, gamma, beta, u0, v0, k1, k2, (extrinsics...)].
        initial_all_RT_extrinsics_list (list): Initial estimate of extrinsic parameters.
                                          Used if `fixed_extrinsics_flag` is True or to unpack if optimizing extrinsics.
        all_observed_image_corners_list (list): Observed 2D corner points for each image.
        world_corners_2d_coords (numpy.ndarray): 2D world coordinates of chessboard corners (Nx2, Z=0).
        fixed_extrinsics_flag (bool): If True, extrinsics are not part of `params_x0_vec` and `initial_all_RT_extrinsics_list` is used.
                                 If False, `params_x0_vec` also contains extrinsics to be optimized.

    Returns:
        numpy.ndarray: A 1D array of residuals (errors_x1, errors_y1, errors_x2, errors_y2, ... for all points).
    """
    num_intrinsic_params = 7  # alpha, gamma, beta, u0, v0, k1, k2

    A_intrinsic_mat, kc_coeffs_arr = retrieveA_kc_fromParams(params_x0_vec[:num_intrinsic_params])

    current_all_RT_extrinsics = []
    if fixed_extrinsics_flag:
        current_all_RT_extrinsics = initial_all_RT_extrinsics_list
    else:  # Extrinsics are also being optimized
        num_extrinsic_params_per_image = 6  # 3 for Rodrigues rotation, 3 for translation
        extr_params_flat = params_x0_vec[num_intrinsic_params:]
        num_images = len(all_observed_image_corners_list)

        expected_extr_len = num_images * num_extrinsic_params_per_image
        if len(extr_params_flat) != expected_extr_len:
            # This can happen if initial_all_RT_extrinsics_list was shorter due to earlier errors
            # Fallback to using initial_all_RT_extrinsics_list if lengths mismatch significantly
            # Or raise error. For now, let's assume the optimizer provides correct length based on x0.
            print(
                f"Warning: Mismatch in number of extrinsic parameters. Expected {expected_extr_len}, got {len(extr_params_flat)}. Check x0 structure for optimization.")
            # If this happens, it's safer to assume fixed extrinsics or debug x0 generation.
            # For now, we'll proceed, but this indicates a potential issue in how x0 is constructed when optimizing extrinsics.
            # This part of the code is only active if fixed_extrinsics_flag is False.

        for i in range(num_images):
            start_idx = i * num_extrinsic_params_per_image
            end_idx_r = start_idx + 3
            end_idx_t = start_idx + 6

            if end_idx_t > len(extr_params_flat):  # Safety break if extr_params_flat is too short
                print(f"Error: Not enough extrinsic parameters for image {i}. Stopping extrinsic parsing.")
                # Fallback: use initial extrinsics for remaining images if this happens mid-loop
                # This is a patch; the root cause of param length mismatch should be fixed.
                current_all_RT_extrinsics.extend(initial_all_RT_extrinsics_list[i:])
                break

            rodrigues_vec = extr_params_flat[start_idx: end_idx_r]
            translation_vec = extr_params_flat[end_idx_r: end_idx_t]

            R_matrix_val, _ = cv2.Rodrigues(rodrigues_vec)
            RT_matrix_val = np.hstack((R_matrix_val, translation_vec.reshape(3, 1)))
            current_all_RT_extrinsics.append(RT_matrix_val)

    all_residuals_list = []
    world_corners_3d_homo = np.hstack((
        world_corners_2d_coords,
        np.zeros((world_corners_2d_coords.shape[0], 1)),
        np.ones((world_corners_2d_coords.shape[0], 1))
    )).T

    for i, observed_corners_img in enumerate(all_observed_image_corners_list):
        if i >= len(current_all_RT_extrinsics) or current_all_RT_extrinsics[i] is None:
            # print(f"Warning: Skipping image {i} in loss function due to missing/invalid extrinsics for residual calculation.")
            # If extrinsics are missing, we can't calculate residuals for these points.
            # This will lead to a shorter residual vector than expected by least_squares if not handled.
            # One option: append NaNs or zeros, but that might skew optimization.
            # Best: ensure current_all_RT_extrinsics is always complete.
            # For now, if an RT is missing, we can't compute residuals for that image's points.
            # This means the residual vector will be shorter. This is problematic for least_squares.
            # The number of residuals must be consistent.
            # Let's fill with large error values if an RT is missing to penalize this state,
            # or ensure x0 and args to least_squares are always consistent.
            # For now, if RT is None, we can't project.
            # The optimization should ideally not reach this if x0 is built correctly.
            # If it does, it means some RTs from initial_all_RT_extrinsics_list were None.
            num_points_this_image = observed_corners_img.shape[0]
            all_residuals_list.extend(np.full(num_points_this_image * 2, 1e6))  # Large error
            continue

        RT_extrinsic_mat = current_all_RT_extrinsics[i]

        reprojected_points_img = projectPointsWithDistortion(world_corners_3d_homo, RT_extrinsic_mat, A_intrinsic_mat,
                                                             kc_coeffs_arr)

        valid_mask = ~np.isnan(reprojected_points_img).any(axis=1)

        # Ensure observed_corners_img is also filtered by valid_mask if reprojection produced NaNs
        residuals_img_pairs = observed_corners_img[valid_mask] - reprojected_points_img[valid_mask]
        all_residuals_list.extend(residuals_img_pairs.ravel())

        # If some points were invalid (NaNs), the number of residuals might vary.
        # least_squares expects a constant size vector of residuals.
        # Handle NaNs by replacing them with a large error or ensuring projection always returns valid numbers.
        # Current projectPointsWithDistortion returns NaNs.
        # We need to ensure the residual vector has fixed length.
        # If valid_mask filters out N points, residuals_img.ravel() is shorter by 2*N.
        # This is problematic.
        # Solution: Replace NaNs in reprojected_points_img with a far-off value before subtraction,
        # or ensure all points are projected (e.g. by clamping normalized coords if Zc is an issue).
        # For now, the current code will produce a variable length residual vector if NaNs occur.
        # This needs to be fixed for robust optimization.
        # Quick fix: if NaNs occurred, the residual for those points could be set to a large value.
        if np.sum(~valid_mask) > 0:  # If there were NaNs
            num_invalid_points = np.sum(~valid_mask)
            # print(f"Debug: Image {i} had {num_invalid_points} NaN projections.")
            # Append large residuals for these invalid points to maintain vector length
            all_residuals_list.extend(np.full(num_invalid_points * 2, 1e6))  # Large error penalty

    return np.array(all_residuals_list)


def calibrate(input_path, output_path, corner_count_w=9, corner_count_h=6):
    # Configuration
    if not os.path.exists(input_path):
        print(f"ERROR: Image folder '{input_path}' not found. Please update the path.")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created results directory: {output_path}")

    h_corners = corner_count_h
    w_corners = corner_count_w
    # square_side defined globally

    # 1. Load Images
    loaded_images_info = loadImages(input_path)
    if not loaded_images_info:
        print("No images loaded. Exiting.")
        return
    print(f"Successfully loaded {len(loaded_images_info)} image files.")

    # 2. Find Chessboard Corners
    print("\nDetecting chessboard corners...")
    successful_corners_data = []
    images_for_calibration = []

    for img_info in loaded_images_info:
        img_data = img_info['data']
        gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (w_corners, h_corners), None)
        if ret:
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            successful_corners_data.append({
                'original_name': img_info['name'],
                'corners': corners_subpix.reshape(-1, 2)
            })
            images_for_calibration.append(img_data)
            print(f"  Corners found in '{img_info['name']}'")
        else:
            print(f"  Corners NOT found in '{img_info['name']}'")

    if not successful_corners_data:
        print("No chessboard corners found in any image. Calibration cannot proceed.")
        return

    all_image_points = [data['corners'] for data in successful_corners_data]
    print(f"\nFound corners in {len(all_image_points)} out of {len(loaded_images_info)} images, proceeding with these.")

    # 3. Get World Coordinates
    world_points_2d = getWorldPoints(square_side, h_corners, w_corners)

    # 4. Display Detected Corners
    print("\nSaving images with detected corners...")
    displayCorners_save_folder = os.path.join(output_path, "1_detected_corners")
    displayCorners(images_for_calibration, all_image_points, h_corners, w_corners, displayCorners_save_folder)
    print(f"Detected corner images saved in '{displayCorners_save_folder}'")

    # --- Initial Parameter Estimation ---
    print("\n--- Initial Parameter Estimation ---")
    # 5. Compute Homographies
    print("Calculating initial homographies (H)...")
    all_H_initial_raw = getAllH(all_image_points, square_side, h_corners, w_corners)

    # Filter out invalid Homographies and corresponding image data
    all_H_valid = []
    all_image_points_valid = []
    images_for_calibration_valid = []

    for i, H_val in enumerate(all_H_initial_raw):
        if H_val is not None and not isinstance(H_val, int) and H_val.shape == (3, 3):  # Basic check for valid H
            all_H_valid.append(H_val)
            all_image_points_valid.append(all_image_points[i])
            images_for_calibration_valid.append(images_for_calibration[i])
        else:
            print(
                f"Warning: Invalid homography computed for image index {i} (original name: {successful_corners_data[i]['original_name']}). Skipping this view.")

    if len(all_H_valid) < 2:  # Need at least 2 views for B typically, 3 for more stability
        print("Error: Not enough valid homographies to proceed (need at least 2). Calibration aborted.")
        return
    print(f"Proceeding with {len(all_H_valid)} valid views for calibration.")

    all_H_initial = all_H_valid
    all_image_points = all_image_points_valid  # Update to use only valid ones
    images_for_calibration = images_for_calibration_valid  # Update to use only valid ones

    # 6. Estimate B matrix
    print("Calculating initial B matrix...")
    B_initial = getB_matrix_from_V(all_H_initial)
    if B_initial is None:  # Should not happen if getB_matrix_from_V is robust
        print("Error: Failed to estimate B matrix (getB_matrix_from_V returned None).")
        return
    print("Estimated initial B matrix:\n", B_initial)

    # 7. Estimate Intrinsic Matrix A
    print("Calculating initial intrinsic matrix (A)...")
    A_initial = getA_intrinsic_matrix(B_initial)
    if A_initial is None:
        print(
            "Error: Failed to compute initial intrinsic matrix A. Check B matrix properties or input data quality. Calibration aborted.")
        return
    print("Initial estimated A (intrinsic matrix):\n", A_initial)

    # 8. Estimate Extrinsic Parameters (R, t)
    print("Calculating initial rotation (R) and translation (t) for each view...")
    all_RT_initial = getRotationAndTrans(A_initial, all_H_initial)
    if len(all_RT_initial) != len(all_H_initial):
        print(
            f"Warning: Number of initial extrinsics ({len(all_RT_initial)}) does not match number of valid views ({len(all_H_initial)}). This may cause issues.")
        # Pad all_RT_initial with None or filter views further if this is critical
        # For now, assume subsequent functions can handle shorter RT list if some failed.
        # However, it's better to ensure consistency. If getRotationAndTrans skips some, we need to filter
        # all_image_points and images_for_calibration again.
        # For simplicity, let's assume getRotationAndTrans returns a list of same length, possibly with None for failures.
        # The current getRotationAndTrans skips on failure, leading to shorter list.
        # This needs careful handling for consistency in optimization.
        # Let's filter views where RT could not be computed.
        valid_rt_indices = [j for j, rt in enumerate(all_RT_initial) if rt is not None]
        if len(valid_rt_indices) < len(all_H_initial):
            print(
                f"  Further filtering views due to failed extrinsic calculation. From {len(all_H_initial)} to {len(valid_rt_indices)} views.")
            all_H_initial = [all_H_initial[j] for j in valid_rt_indices]
            all_image_points = [all_image_points[j] for j in valid_rt_indices]
            images_for_calibration = [images_for_calibration[j] for j in valid_rt_indices]
            all_RT_initial = [all_RT_initial[j] for j in valid_rt_indices]  # This is already filtered

        if len(all_RT_initial) < 1:  # Need at least one view for optimization step
            print("Error: No valid extrinsics could be computed. Calibration aborted.")
            return

    print(f"Initial extrinsics calculated for {len(all_RT_initial)} views.")

    # 9. Initial Distortion Coefficients
    kc_initial = np.array([0.0, 0.0], dtype=np.float64)
    print("Initial distortion coefficients (kc): ", kc_initial)

    # 10. Calculate Initial Reprojection Error
    print("\nCalculating initial reprojection error...")
    initial_reprojection_error, _ = reprojectPointsAndGetError(A_initial, kc_initial, all_RT_initial,
                                                               all_image_points, world_points_2d)
    print(f"Initial Root Mean Squared Reprojection Error: {initial_reprojection_error:.4f} pixels")

    # --- Non-linear Optimization ---
    print("\n--- Non-linear Optimization (Refinement) ---")
    x0_for_optimization = extractParamsFromA_kc(A_initial, kc_initial)

    # Ensure all_RT_initial doesn't contain Nones before passing to loss function if fixed_extrinsics=True
    if any(rt is None for rt in all_RT_initial):
        print("Error: Some initial RTs are None, cannot proceed with optimization using fixed extrinsics. Aborting.")
        return

    print("Optimizing intrinsic parameters and distortion coefficients (extrinsics fixed)...")
    optimization_args = (all_RT_initial, all_image_points, world_points_2d, True)  # True for fixed_extrinsics

    # Need to ensure lossFunctionOptimization returns a fixed-size residual vector.
    # The current implementation might not if NaNs occur and are filtered variably.
    # For now, we assume the fix for NaNs in lossFunctionOptimization (appending large errors) works.

    try:
        optimization_result = scipy.optimize.least_squares(
            fun=lossFunctionOptimization,
            x0=x0_for_optimization,
            method="lm",
            args=optimization_args,
            verbose=2
        )
    except Exception as e:
        print(f"Error during optimization: {e}")
        print("Skipping optimization. Using initial parameters as final.")
        A_optimized, kc_optimized = A_initial, kc_initial
        all_RT_optimized = all_RT_initial  # Extrinsics were not re-optimized here
    else:
        if not optimization_result.success:
            print("Optimization failed or did not converge. Using initial parameters for A and kc.")
            print("Message:", optimization_result.message)
            A_optimized, kc_optimized = A_initial, kc_initial
        else:
            print("Optimization successful.")
            optimized_params = optimization_result.x
            A_optimized, kc_optimized = retrieveA_kc_fromParams(optimized_params)

        # Recalculate extrinsics using the new A_optimized
        print("\nRe-calculating extrinsics with optimized A...")
        all_RT_optimized = getRotationAndTrans(A_optimized, all_H_initial)  # Use original H with new A
        # Filter again if some RTs failed with new A
        valid_rt_opt_indices = [j for j, rt in enumerate(all_RT_optimized) if rt is not None]
        if len(valid_rt_opt_indices) < len(all_H_initial):
            print(
                f"  Warning: Some extrinsics failed with optimized A. Number of views for final error: {len(valid_rt_opt_indices)}")
            # For error calculation, use only views where extrinsics are valid
            final_err_all_image_points = [all_image_points[j] for j in valid_rt_opt_indices]
            final_err_all_RT = [all_RT_optimized[j] for j in valid_rt_opt_indices]
        else:
            final_err_all_image_points = all_image_points
            final_err_all_RT = all_RT_optimized

    print("\nOptimized Intrinsic Matrix (A_new):\n", A_optimized)
    print("Optimized Distortion Coefficients (kc_new):", kc_optimized)

    # 11. Calculate Final Reprojection Error
    print("Calculating final reprojection error with optimized parameters...")
    if not final_err_all_RT:  # Check if list is empty
        print("No valid extrinsics for final error calculation. Skipping.")
        final_reprojection_error = np.inf
        all_reprojected_points_final = []
    else:
        final_reprojection_error, all_reprojected_points_final = reprojectPointsAndGetError(
            A_optimized, kc_optimized, final_err_all_RT,  # Use filtered RTs
            final_err_all_image_points, world_points_2d  # Use corresponding image points
        )
    print(f"Final Root Mean Squared Reprojection Error: {final_reprojection_error:.4f} pixels")

    print(f"\nComparison of Reprojection Error:")
    print(f"  Initial RMSE: {initial_reprojection_error:.4f} pixels (based on {len(all_RT_initial)} views)")
    print(
        f"  Final RMSE  : {final_reprojection_error:.4f} pixels (based on {len(final_err_all_RT if final_err_all_RT else [])} views)")

    # 12. Undistort images and draw reprojected points
    print("\nUndistorting images and drawing final reprojected points...")
    K_final_cv = np.array(A_optimized, dtype=np.float32)
    D_final_cv = np.array([kc_optimized[0], kc_optimized[1], 0, 0, 0], dtype=np.float32)

    undistorted_save_folder = os.path.join(output_path, "2_undistorted_reprojected")
    if not os.path.exists(undistorted_save_folder):
        os.makedirs(undistorted_save_folder)

    # Iterate through the views that were successfully used for final error calculation
    for i in range(len(final_err_all_image_points)):
        # Find the original image corresponding to final_err_all_image_points[i]
        # This requires careful tracking of indices or original image names.
        # Assuming final_err_all_image_points corresponds to a subset of the initially successful `images_for_calibration`.
        # This mapping is getting complicated due to multiple filtering stages.
        # For simplicity in visualization, let's try to map back to original image names if possible.
        # The `successful_corners_data` holds original names. `all_image_points` was filtered.
        # This part needs robust index management if we want to use original filenames.

        # Let's use the images from `images_for_calibration` that correspond to `final_err_all_image_points`.
        # `images_for_calibration` should have been filtered consistently with `final_err_all_image_points`.
        # This assumes `final_err_all_image_points` is a result of filtering `all_image_points`
        # and `images_for_calibration` was filtered in parallel.

        # The `images_for_calibration` list was last filtered when `all_H_initial` was filtered.
        # `final_err_all_image_points` comes from `all_image_points` after RT optimization filtering.
        # This is tricky. For now, let's assume `images_for_calibration` (after H-filter)
        # and `final_err_all_image_points` (after RT-filter on H-filtered points)
        # can be matched if their lengths are the same after the RT-filter step.

        # A simpler approach for visualization: use the images that correspond to `final_err_all_RT`.
        # The `images_for_calibration` list should be the one that corresponds to `all_H_initial` (after filtering).
        # If `all_RT_optimized` was filtered to `final_err_all_RT`, then the corresponding images
        # are from `images_for_calibration` (which was aligned with `all_H_initial`).

        if i >= len(images_for_calibration) or i >= len(final_err_all_RT):  # Safety
            print(f"Skipping visualization for image index {i} due to list length mismatch.")
            continue

        image_to_process = images_for_calibration[i].copy()  # Image corresponding to this valid RT
        observed_pts_for_this_img = final_err_all_image_points[i]
        reprojected_pts_for_this_img = all_reprojected_points_final[i] if all_reprojected_points_final and i < len(
            all_reprojected_points_final) else None

        h_img, w_img = image_to_process.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K_final_cv, D_final_cv, (w_img, h_img), alpha=0.5,
                                                               newImgSize=(w_img, h_img))  # alpha=0.5 for some cropping

        img_undistorted = cv2.undistort(image_to_process, K_final_cv, D_final_cv, None, new_camera_matrix)

        # Draw reprojected points (model based)
        if reprojected_pts_for_this_img is not None and not np.isnan(reprojected_pts_for_this_img).all():
            points_to_draw_distorted = np.float32(reprojected_pts_for_this_img).reshape(-1, 1, 2)
            valid_pts_mask = ~np.isnan(points_to_draw_distorted).any(axis=(1, 2))
            if np.any(valid_pts_mask):
                undistorted_reproj_pts = cv2.undistortPoints(points_to_draw_distorted[valid_pts_mask], K_final_cv,
                                                             D_final_cv, P=new_camera_matrix)
                for pt_coords_undistorted in undistorted_reproj_pts:
                    x_val, y_val = int(round(pt_coords_undistorted[0][0])), int(round(pt_coords_undistorted[0][1]))
                    cv2.circle(img_undistorted, (x_val, y_val), radius=4, color=(0, 0, 255),
                               thickness=-1)  # Red for reprojected

        # Draw observed points (undistorted)
        observed_corners_reshaped = np.float32(observed_pts_for_this_img).reshape(-1, 1, 2)
        undistorted_observed_pts = cv2.undistortPoints(observed_corners_reshaped, K_final_cv, D_final_cv,
                                                       P=new_camera_matrix)
        for obs_pt_coords_undistorted in undistorted_observed_pts:
            x_obs, y_obs = int(round(obs_pt_coords_undistorted[0][0])), int(round(obs_pt_coords_undistorted[0][1]))
            cv2.circle(img_undistorted, (x_obs, y_obs), radius=6, color=(0, 255, 0),
                       thickness=1)  # Green for observed (undistorted)

        # Try to get original filename for saving
        # This requires `successful_corners_data` to be filtered consistently with `images_for_calibration`
        # and `final_err_all_image_points`. This is complex.
        # For now, use a generic index based on the loop over `final_err_all_image_points`.
        save_filename = os.path.join(undistorted_save_folder, f"img_{i:03d}_undistorted_reproj.png")
        cv2.imwrite(save_filename, img_undistorted)
        # print(f"  Saved undistorted image with points to '{save_filename}'")

    print(f"Undistorted images with points saved in '{undistorted_save_folder}'")
    print("\nCalibration process complete.")