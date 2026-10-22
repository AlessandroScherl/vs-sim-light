# The two functions of the DINOBot gist that the port reproduces, copied byte-for-byte
# from the vendored gist beside this file (`dinobot_oneshot_il.py`, final revision
# 60d5adbea956d36d6a2d039a59f9524875b8aa0c of 2024-05-21, lines 87-107 and 109-110),
# because the gist itself does not parse (typographic quotes in its `__main__` guard,
# two `def` lines without a colon) and cannot be imported. `numpy as np` is the gist's
# own line 10. Do not edit: tests/fast/test_dinobot.py runs these beside
# `vssim.baselines.dinobot.kabsch` / `compute_error`.
import numpy as np 


def find_transformation(X, Y):
    """
    Inputs: X, Y: lists of 3D points
    Outputs: R - 3x3 rotation matrix, t - 3-dim translation array.
    Find transformation given two sets of correspondences between 3D points.
    """
    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t

def compute_error(points1, points2):
    return np.linalg.norm(np.array(points1) - np.array(points2))
