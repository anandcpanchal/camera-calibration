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


def getAllH(all_corners, square_side, h, w):
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
    set1_world_points = getWorldPoints(square_side, h, w)  # 3D world points (Z=0)
    all_H = []
    for corners_image_points in all_corners:  # For each image
        set2_image_points = corners_image_points  # 2D image points
        H = getH(set1_world_points, set2_image_points)  # Compute homography
        # Alternative using OpenCV's findHomography (often more robust due to RANSAC)
        # H, _ = cv2.findHomography(set1_world_points, set2_image_points, cv2.RANSAC, 5.0)
        all_H.append(H)
    return all_H


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


def getV(all_H):
    """
    Construct the matrix V_constraints_matrix from all homography matrices.
    Each homography provides two constraints for V_constraints_matrix*b = 0.

    Args:
        all_H (list of numpy.ndarray): A list of 3x3 homography matrices.

    Returns:
        numpy.ndarray: The matrix V_constraints_matrix, where each row is a constraint.
    """
    v_constraints = []
    for H_matrix in all_H:
        h1 = H_matrix[:, 0]  # First column of H
        h2 = H_matrix[:, 1]  # Second column of H
        # h3 = H_matrix[:, 2] # Third column of H (not used for these constraints)

        # Constraint from h1^T B h2 = 0  => v12^T b = 0
        v12 = getVij(h1, h2)
        # Constraint from h1^T B h1 = h2^T B h2 => (v11 - v22)^T b = 0
        v11 = getVij(h1, h1)
        v22 = getVij(h2, h2)

        v_constraints.append(v12.T)
        v_constraints.append((v11 - v22).T)
    return np.array(v_constraints)


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


def getB_matrix_from_V(all_H):
    """
    Estimate the matrix B_matrix (related to the camera intrinsics A_intrinsic by B_matrix = lambda_val_scalar * A_intrinsic^-T A_intrinsic^-1)
    by solving V_constraints_matrix*b = 0.

    Args:
        all_H (list of numpy.ndarray): A list of 3x3 homography matrices.

    Returns:
        numpy.ndarray: The estimated 3x3 symmetric matrix B_matrix.
    """
    V_constraints_matrix = getV(all_H)  # Construct the V_constraints_matrix matrix from constraints
    # Solve V_constraints_matrix*b = 0 using SVD. The solution b is the eigenvector corresponding
    # to the smallest eigenvalue of V_constraints_matrix^T V_constraints_matrix, which is the last column of V_svd_V_transpose.
    U_svd, sigma_svd, V_svd_V_transpose = np.linalg.svd(V_constraints_matrix)
    b_solution = V_svd_V_transpose[-1, :]  # The last row of V_svd_V_transpose (or last column of V_svd_V)
    print("b vector (solution to Vb=0): ", b_solution)
    B_matrix_val = arrangeB_symmetric(b_solution)  # Arrange b into the 3x3 matrix B_matrix_val
    return B_matrix_val


def getA_intrinsic_matrix(B_matrix_input):
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
    # Extract elements of B for clarity, following common notation B_ij
    # B = [[B11, B12, B13],
    #      [B12, B22, B23],
    #      [B13, B23, B33]]
    # B_matrix_input[0,0] = B11, B_matrix_input[0,1] = B12, B_matrix_input[0,2] = B13
    # B_matrix_input[1,1] = B22, B_matrix_input[1,2] = B23
    # B_matrix_input[2,2] = B33
    B11, B12, B13 = B_matrix_input[0, 0], B_matrix_input[0, 1], B_matrix_input[0, 2]
    B22, B23 = B_matrix_input[1, 1], B_matrix_input[1, 2]  # B_matrix_input[1,0] is B12
    B33 = B_matrix_input[2, 2]

    v0_denominator = B11 * B22 - B12 ** 2

    # Principal point y-coordinate (v0 or cy)
    v0 = (B12 * B13 - B11 * B23) / v0_denominator

    # Lambda (overall scale factor, must be positive)
    # lambda_val_scalar = B33 - (B13^2 + v0 * (B12 * B13 - B11 * B23)) / B11
    lambda_val_numerator_term = B13 ** 2 + v0 * (B12 * B13 - B11 * B23)
    lambda_val_scalar = B33 - (lambda_val_numerator_term / B11)  # B11 already checked > 0

    # Focal length in x-direction (alpha or fx)
    alpha_sq_arg = lambda_val_scalar / B11
    # Argument should be positive if lambda_val_scalar > 0 and B11 > 0
    alpha = np.sqrt(alpha_sq_arg)

    # Focal length in y-direction (beta or fy)
    beta_sq_arg = (lambda_val_scalar * B11) / v0_denominator

    beta = np.sqrt(beta_sq_arg)

    # Skew factor (gamma or s)
    # lambda_val_scalar is confirmed positive here.
    gamma = - (B12 * alpha ** 2 * beta) / lambda_val_scalar

    # Principal point x-coordinate (u0 or cx)
    # beta is confirmed positive (or rather, beta_sq_arg > 1e-9, so beta > some_epsilon).
    # lambda_val_scalar is confirmed positive.
    u0 = (gamma * v0 / beta) - (B13 * alpha ** 2 / lambda_val_scalar)

    A_intrinsic_matrix = np.array([
        [alpha, gamma, u0],
        [0.0, beta, v0],  # Use 0.0 for float consistency
        [0.0, 0.0, 1.0]  # Use 0.0, 1.0 for float consistency
    ], dtype=np.float64)  # Explicitly set dtype for the whole matrix

    return A_intrinsic_matrix


def getRotationAndTrans(A_intrinsic, all_H):
    """
    Compute the rotation (R_matrix_val) and translation (t_vec) [R_matrix_val|t_vec] for each view (homography).

    Args:
        A_intrinsic (numpy.ndarray): The 3x3 intrinsic camera matrix.
        all_H (list of numpy.ndarray): A list of 3x3 homography matrices.

    Returns:
        list of numpy.ndarray: A list of 3x4 [R_matrix_val|t_vec] extrinsic parameter matrices.
    """
    all_RT = []
    if A_intrinsic is None:  # Guard against None A_intrinsic
        print("Error in getRotationAndTrans: Intrinsic matrix A is None.")
        return all_RT  # Return empty list or handle error appropriately

    A_inv = np.linalg.inv(A_intrinsic)

    for H_matrix in all_H:
        if H_matrix is None or isinstance(H_matrix, int):  # Check if H is valid
            print("Warning: Invalid H matrix encountered in getRotationAndTrans. Skipping.")
            # Optionally append a placeholder or continue
            # For now, let's append a dummy RT to maintain list length if needed by caller,
            # or simply skip and the caller handles varying list lengths.
            # To be safe, let's make sure the caller can handle shorter all_RT lists if H is bad.
            # For now, if H is bad, we just skip this iteration.
            continue

        h1 = H_matrix[:, 0]  # First column of H
        h2 = H_matrix[:, 1]  # Second column of H
        h3 = H_matrix[:, 2]  # Third column of H (translation part)

        norm_A_inv_h1 = np.linalg.norm(np.dot(A_inv, h1))
        if norm_A_inv_h1 < 1e-9:  # Avoid division by zero or very small numbers
            print("Warning: Norm of A_inv * h1 is close to zero. Extrinsics may be ill-defined for an H matrix.")
            # Create a dummy RT or skip. Let's skip for now.
            # RT_matrix_val = np.hstack((np.eye(3), np.zeros((3,1))))
            # all_RT.append(RT_matrix_val)
            continue

        r1 = np.dot(A_inv, h1) / norm_A_inv_h1
        r2 = np.dot(A_inv, h2) / norm_A_inv_h1
        # r3 must be orthogonal to r1 and r2. Ensure r1 and r2 are not collinear.
        # This is implicitly handled if H is from a good calibration pattern.
        r3 = np.cross(r1, r2)
        t_vec = np.dot(A_inv, h3) / norm_A_inv_h1

        # Form the rotation matrix R_matrix_val = [r1, r2, r3]
        R_matrix_val = np.column_stack((r1, r2, r3))

        # Re-orthogonalize R_matrix_val to ensure it's a valid rotation matrix (e.g., using SVD)
        # This is a good practice as numerical errors can make R_matrix_val not perfectly orthogonal.
        U_r, _, Vt_r = np.linalg.svd(R_matrix_val)
        R_orthogonal = np.dot(U_r, Vt_r)
        # Ensure R_orthogonal has determinant +1 (a proper rotation matrix)
        if np.linalg.det(R_orthogonal) < 0:
            Vt_r[-1, :] *= -1  # Flip the sign of the last row of Vt_r (or last column of U_r)
            R_orthogonal = np.dot(U_r, Vt_r)

        # Use the orthogonalized rotation matrix
        RT_matrix_val = np.hstack((R_orthogonal, t_vec.reshape(3, 1)))  # Combine R_orthogonal and t_vec
        all_RT.append(RT_matrix_val)

    return all_RT

