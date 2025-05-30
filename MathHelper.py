import numpy as np
import cv2


def getWorldPoints(square_side, h, w):
    """
    Generate 3D world coordinates of the chessboard corners.
    The chessboard is assumed to be on the Z=0 plane.

    Args:
        square_side (float): The side length of a square on the chessboard.
        h (int): Number of internal corners along the height of the chessboard.
        w (int): Number of internal corners along the width of the chessboard.

    Returns:
        numpy.ndarray: An array of shape (h*w, 2) containing the (X, Y) coordinates of the world points.
                       Z is assumed to be 0 for all points.
    """
    # h, w = [6, 9] # Example dimensions
    Yi, Xi = np.indices((h, w))  # Create 2D arrays of Y and X indices
    offset = 0  # Offset can be used if the origin is not at the corner
    # Create a stack of (X, Y) coordinates, scaled by square_side
    lin_homg_pts = np.stack(((Xi.ravel() + offset) * square_side, (Yi.ravel() + offset) * square_side)).T
    return lin_homg_pts


def getH(set1, set2):
    """
    Compute the homography matrix H from two sets of corresponding points (set1 -> H * set2).
    This uses the Direct Linear Transform (DLT) algorithm with SVD.

    Args:
        set1 (numpy.ndarray): First set of 2D points (N x 2).
        set2 (numpy.ndarray): Second set of 2D points (N x 2), corresponding to set1.

    Returns:
        numpy.ndarray: The 3x3 homography matrix, normalized such that H[2,2] = 1.
                       Returns 0 if fewer than 4 points are provided.
    """
    nrows = set1.shape[0]
    if (nrows < 4):
        print("Need at least four points to compute SVD for homography.")
        return 0

    x = set1[:, 0]  # X coordinates of the first set
    y = set1[:, 1]  # Y coordinates of the first set
    xp = set2[:, 0]  # X coordinates of the second set (image points)
    yp = set2[:, 1]  # Y coordinates of the second set (image points)

    A_matrix_构造 = []  # Matrix A_matrix_构造 for the linear system Ah = 0
    for i in range(nrows):
        # For each correspondence, two rows are added to A_matrix_构造
        row1 = np.array([x[i], y[i], 1, 0, 0, 0, -x[i] * xp[i], -y[i] * xp[i], -xp[i]])
        A_matrix_构造.append(row1)
        row2 = np.array([0, 0, 0, x[i], y[i], 1, -x[i] * yp[i], -y[i] * yp[i], -yp[i]])
        A_matrix_构造.append(row2)

    A_matrix_构造 = np.array(A_matrix_构造)
    U, E, V = np.linalg.svd(A_matrix_构造, full_matrices=True)  # Singular Value Decomposition
    H = V[-1, :].reshape((3, 3))  # The solution h is the last column of V (or last row of V.T)
    H = H / H[2, 2]  # Normalize H so that H[2,2] = 1
    return H


def getAllH(all_corners, square_side, h, w, logger):
    """
    Compute homography matrices for all images.

    Args:
        all_corners (list of numpy.ndarray): A list where each element is an array of detected
                                             2D image corners for one calibration image.
        square_side (float): The side length of a square on the chessboard.
        h (int): Number of internal corners along the height.
        w (int): Number of internal corners along the width.

    Returns:
        list of numpy.ndarray: A list of 3x3 homography matrices, one for each image.
    """
    if logger: logger.debug(
        f"Entering getAllH. Number of corner sets: {len(all_corners)}, square_side: {square_side}, h: {h}, w: {w}")
    set1_world_points = getWorldPoints(square_side, h, w)
    if logger: logger.debug(
        f"Generated world points for homography: shape {set1_world_points.shape}, first few points:\n{set1_world_points[:3]}")

    all_H_list = []
    for i, corners_image_points in enumerate(all_corners):
        if logger: logger.debug(
            f"Processing corner set {i + 1}/{len(all_corners)} for homography. Image points shape: {corners_image_points.shape}")
        H_matrix = getH(set1_world_points, corners_image_points)
        if H_matrix is None or isinstance(H_matrix, int):  # getH returns 0 or None on failure
            if logger: logger.warning(f"Failed to compute homography for corner set {i}. Result: {H_matrix}")
            all_H_list.append(None)  # Append None to mark failure for this view
        else:
            if logger: logger.debug(f"Computed homography for set {i}:\n{H_matrix}")
            all_H_list.append(H_matrix)
    if logger: logger.debug("Exiting getAllH.")
    return all_H_list


def getVij(hi, hj):
    """
    Compute the vector vij from two columns of the homography matrix H.
    This is used to form constraints on the image of the absolute conic.

    Args:
        hi (numpy.ndarray): The i-th column of H (as a 1D array or column vector).
        hj (numpy.ndarray): The j-th column of H (as a 1D array or column vector).

    Returns:
        numpy.ndarray: The 6-element vector vij.
                       vij = [hi0*hj0, hi0*hj1 + hi1*hj0, hi1*hj1,
                              hi2*hj0 + hi0*hj2, hi2*hj1 + hi1*hj2, hi2*hj2]^T
    """
    Vij = np.array([hi[0] * hj[0],
                    hi[0] * hj[1] + hi[1] * hj[0],
                    hi[1] * hj[1],
                    hi[2] * hj[0] + hi[0] * hj[2],
                    hi[2] * hj[1] + hi[1] * hj[2],
                    hi[2] * hj[2]])
    return Vij.T  # Return as a column vector (or a 1D array that behaves like one in dot products)


def getV(all_H, logger):
    """
    Construct the matrix V_constraints_matrix from all homography matrices.
    Each homography provides two constraints for V_constraints_matrix*b = 0.

    Args:
        all_H (list of numpy.ndarray): A list of 3x3 homography matrices.

    Returns:
        numpy.ndarray: The matrix V_constraints_matrix, where each row is a constraint.
    """
    if logger: logger.debug(f"Entering getV. Number of H matrices: {len(all_H)}")
    v_constraints = []
    for i, H_matrix in enumerate(all_H):
        if H_matrix is None:
            if logger: logger.warning(f"Skipping H matrix at index {i} in getV because it's None.")
            continue  # Skip if homography computation failed for this view

        h1 = H_matrix[:, 0]
        h2 = H_matrix[:, 1]

        v12 = getVij(h1, h2)
        v11 = getVij(h1, h1)
        v22 = getVij(h2, h2)

        v_constraints.append(v12.T)
        v_constraints.append((v11 - v22).T)
        if logger: logger.debug(f"For H matrix {i}, added v12 and (v11-v22) constraints.")

    if not v_constraints:  # If no valid H matrices were processed
        if logger: logger.error("No valid constraints generated in getV. Returning empty array.")
        return np.array([])

    result_v_matrix = np.array(v_constraints)
    if logger: logger.debug(f"Exiting getV. Constructed V matrix shape: {result_v_matrix.shape}")
    return result_v_matrix


def arrangeB_symmetric(b_vector):
    """
    Arrange the 6-element vector b into a symmetric 3x3 matrix B_matrix.
    B_matrix = [[b0, b1, b3],
         [b1, b2, b4],
         [b3, b4, b5]]

    Args:
        b_vector (numpy.ndarray): The 6-element vector b = [B11, B12, B22, B13, B23, B33].

    Returns:
        numpy.ndarray: The 3x3 symmetric matrix B_matrix.
    """
    B_mat = np.zeros((3, 3))
    B_mat[0, 0] = b_vector[0]  # B11
    B_mat[0, 1] = b_vector[1]  # B12
    B_mat[0, 2] = b_vector[3]  # B13
    B_mat[1, 0] = b_vector[1]  # B21 = B12
    B_mat[1, 1] = b_vector[2]  # B22
    B_mat[1, 2] = b_vector[4]  # B23
    B_mat[2, 0] = b_vector[3]  # B31 = B13
    B_mat[2, 1] = b_vector[4]  # B32 = B23
    B_mat[2, 2] = b_vector[5]  # B33
    return B_mat


def getB_matrix_from_V(all_H, logger):
    """
    Estimate the matrix B_matrix (related to the camera intrinsics A_intrinsic by B_matrix = lambda_val_scalar * A_intrinsic^-T A_intrinsic^-1)
    by solving V_constraints_matrix*b = 0.

    Args:
        all_H (list of numpy.ndarray): A list of 3x3 homography matrices.

    Returns:
        numpy.ndarray: The estimated 3x3 symmetric matrix B_matrix.
    """
    if logger: logger.debug("Entering getB_matrix_from_V.")
    V_constraints_matrix = getV(all_H, logger=logger)
    if V_constraints_matrix.size == 0 or V_constraints_matrix.shape[0] < 1:  # Check if V is empty or has too few rows
        if logger: logger.error("V matrix is empty or has insufficient constraints. Cannot compute B.")
        return None  # Cannot compute B if V is empty

    if logger: logger.debug(f"V matrix for SVD (shape {V_constraints_matrix.shape}):\n{V_constraints_matrix}")

    try:
        U_svd, sigma_svd, V_svd_V_transpose = np.linalg.svd(V_constraints_matrix)
    except np.linalg.LinAlgError as e:
        if logger: logger.error(f"SVD computation for V matrix failed: {e}", exc_info=True)
        return None

    b_solution = V_svd_V_transpose[-1, :]
    if logger: logger.debug(f"b vector (solution to Vb=0 from SVD of V, shape {b_solution.shape}): {b_solution}")

    B_matrix_val = arrangeB_symmetric(b_solution)
    if logger: logger.debug(f"Arranged B matrix:\n{B_matrix_val}")
    if logger: logger.debug("Exiting getB_matrix_from_V.")
    return B_matrix_val


def getA_intrinsic_matrix(B_matrix_input, logger):
    """
    Compute the intrinsic camera matrix A_intrinsic from the matrix B_matrix_input.
    A_intrinsic = [[alpha, gamma, u0],
         [0,     beta,  v0],
         [0,     0,     1 ]]
    B_matrix_input = lambda_val_scalar * A_intrinsic^-T A_intrinsic^-1

    Args:
        B_matrix_input (numpy.ndarray): The 3x3 symmetric matrix B_matrix_input.

    Returns:
        numpy.ndarray: The 3x3 upper-triangular intrinsic camera matrix A_intrinsic.
                       Returns None if B_matrix_input is not suitable (e.g., not positive definite).
    """
    if logger: logger.debug(f"Entering getA_intrinsic_matrix with B_matrix_input:\n{B_matrix_input}")

    if B_matrix_input is None:
        if logger: logger.error("B_matrix_input is None. Cannot compute A.")
        return None

    B11, B12, B13 = B_matrix_input[0, 0], B_matrix_input[0, 1], B_matrix_input[0, 2]
    B22, B23 = B_matrix_input[1, 1], B_matrix_input[1, 2]
    B33 = B_matrix_input[2, 2]
    if logger: logger.debug(
        f"B matrix elements: B11={B11:.4e}, B12={B12:.4e}, B13={B13:.4e}, B22={B22:.4e}, B23={B23:.4e}, B33={B33:.4e}")

    v0_denominator = B11 * B22 - B12 ** 2
    if logger: logger.debug(f"v0_denominator: {v0_denominator:.4e}")

    v0 = (B12 * B13 - B11 * B23) / v0_denominator
    if logger: logger.debug(f"Calculated v0: {v0:.4e}")

    lambda_val_numerator_term = B13 ** 2 + v0 * (B12 * B13 - B11 * B23)
    lambda_val_scalar = B33 - (lambda_val_numerator_term / B11)
    if logger: logger.debug(f"Calculated lambda_val_scalar: {lambda_val_scalar:.4e}")

    alpha_sq_arg = lambda_val_scalar / B11
    alpha = np.sqrt(alpha_sq_arg)
    if logger: logger.debug(f"Calculated alpha: {alpha:.4e} (from alpha_sq_arg: {alpha_sq_arg:.4e})")

    beta_sq_arg = (lambda_val_scalar * B11) / v0_denominator
    beta = np.sqrt(beta_sq_arg)
    if logger: logger.debug(f"Calculated beta: {beta:.4e} (from beta_sq_arg: {beta_sq_arg:.4e})")

    gamma = - (B12 * alpha ** 2 * beta) / lambda_val_scalar
    if logger: logger.debug(f"Calculated gamma (skew): {gamma:.4e}")

    u0 = (gamma * v0 / beta) - (B13 * alpha ** 2 / lambda_val_scalar)
    if logger: logger.debug(f"Calculated u0: {u0:.4e}")

    A_intrinsic_matrix = np.array([
        [alpha, gamma, u0],
        [0.0, beta, v0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    if logger: logger.debug(f"Constructed A_intrinsic_matrix:\n{A_intrinsic_matrix}")
    if logger: logger.debug("Exiting getA_intrinsic_matrix.")
    return A_intrinsic_matrix


def getRotationAndTrans(A_intrinsic, all_H, logger):
    """
    Compute the rotation (R_matrix_val) and translation (t_vec) [R_matrix_val|t_vec] for each view (homography).

    Args:
        A_intrinsic (numpy.ndarray): The 3x3 intrinsic camera matrix.
        all_H (list of numpy.ndarray): A list of 3x3 homography matrices.

    Returns:
        list of numpy.ndarray: A list of 3x4 [R_matrix_val|t_vec] extrinsic parameter matrices.
    """
    if logger: logger.debug(
        f"Entering getRotationAndTrans. A_intrinsic:\n{A_intrinsic}\nNumber of H matrices: {len(all_H)}")
    all_RT_list = []
    if A_intrinsic is None:
        if logger: logger.error("Intrinsic matrix A is None. Cannot compute extrinsics.")
        # Return a list of Nones matching all_H length to maintain correspondence if caller expects it
        return [None] * len(all_H)

    try:
        A_inv = np.linalg.inv(A_intrinsic)
        if logger: logger.debug(f"Inverse of A_intrinsic:\n{A_inv}")
    except np.linalg.LinAlgError as e:
        if logger: logger.error(f"Failed to invert A_intrinsic: {e}", exc_info=True)
        return [None] * len(all_H)

    for i, H_matrix in enumerate(all_H):
        if H_matrix is None:
            if logger: logger.warning(f"Skipping H matrix at index {i} in getRotationAndTrans because it's None.")
            all_RT_list.append(None)  # Maintain list structure
            continue

        if logger: logger.debug(f"Processing H matrix {i + 1}/{len(all_H)} for extrinsics:\n{H_matrix}")
        h1 = H_matrix[:, 0]
        h2 = H_matrix[:, 1]
        h3 = H_matrix[:, 2]

        norm_A_inv_h1 = np.linalg.norm(np.dot(A_inv, h1))
        if logger: logger.debug(f"Norm of A_inv * h1 for H {i}: {norm_A_inv_h1:.4e}")

        if norm_A_inv_h1 < 1e-9:
            if logger: logger.warning(f"Norm of A_inv * h1 is close to zero for H {i}. Extrinsics may be ill-defined.")
            all_RT_list.append(None)
            continue

        r1 = np.dot(A_inv, h1) / norm_A_inv_h1
        r2 = np.dot(A_inv, h2) / norm_A_inv_h1
        r3 = np.cross(r1, r2)
        t_vec = np.dot(A_inv, h3) / norm_A_inv_h1
        if logger: logger.debug(f"Raw r1, r2, r3, t for H {i}:\nr1={r1}\nr2={r2}\nr3={r3}\nt={t_vec}")

        R_matrix_val = np.column_stack((r1, r2, r3))
        if logger: logger.debug(f"Constructed R matrix (before orthogonalization) for H {i}:\n{R_matrix_val}")

        try:
            U_r, _, Vt_r = np.linalg.svd(R_matrix_val)
            R_orthogonal = np.dot(U_r, Vt_r)
            if np.linalg.det(R_orthogonal) < 0:
                if logger: logger.debug(
                    f"Determinant of R for H {i} is negative ({np.linalg.det(R_orthogonal):.4f}), flipping sign of Vt_r's last row.")
                Vt_r[-1, :] *= -1
                R_orthogonal = np.dot(U_r, Vt_r)
            if logger: logger.debug(f"Orthogonalized R matrix for H {i}:\n{R_orthogonal}")
        except np.linalg.LinAlgError as e:
            if logger: logger.error(f"SVD for R matrix orthogonalization failed for H {i}: {e}", exc_info=True)
            all_RT_list.append(None)
            continue

        RT_matrix_val = np.hstack((R_orthogonal, t_vec.reshape(3, 1)))
        if logger: logger.debug(f"Final [R|t] matrix for H {i}:\n{RT_matrix_val}")
        all_RT_list.append(RT_matrix_val)

    if logger: logger.debug("Exiting getRotationAndTrans.")
    return all_RT_list

