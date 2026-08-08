import math
import random
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def extract_logo_points(image_path, target_width, target_height, num_points):
    """
    Load a logo image, convert to binary, and sample exactly num_points.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return np.zeros((num_points, 2))
        
    # Resize to fit in target bounding box
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Extract alpha channel or convert to grayscale
    if img.shape[2] == 4:
        gray = img[:, :, 3] # Use alpha channel for mask
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray) # Assuming white background
        
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Get coordinates of non-zero pixels
    y_coords, x_coords = np.nonzero(mask)
    pts = np.column_stack((x_coords, y_coords))
    
    if len(pts) == 0:
        return np.zeros((num_points, 2))
        
    # Randomly sample num_points
    if len(pts) >= num_points:
        indices = np.random.choice(len(pts), num_points, replace=False)
    else:
        indices = np.random.choice(len(pts), num_points, replace=True)
        
    sampled = pts[indices].astype(np.float32)
    
    # Center the logo in the bounding box
    min_x, max_x = np.min(sampled[:, 0]), np.max(sampled[:, 0])
    min_y, max_y = np.min(sampled[:, 1]), np.max(sampled[:, 1])
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    
    sampled[:, 0] += (target_width / 2) - cx
    sampled[:, 1] += (target_height / 2) - cy
    
    return sampled


def optimal_transport(src_pts, dst_pts):
    """
    Map src_pts to dst_pts such that total squared distance is minimized.
    Returns reordered dst_pts.
    """
    # Subsample if too large to avoid memory issues with O(N^2) distance matrix
    # But for N=900, 900x900 is 810,000 floats, which is trivial.
    cost_matrix = cdist(src_pts, dst_pts, metric='sqeuclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return dst_pts[col_ind]


def compute_traveller_routes(portrait_pts, logo_paths, target_w, target_h, num_points):
    """
    Computes the optimal transport cycle for the travellers.
    """
    logos = []
    for path in logo_paths:
        pts = extract_logo_points(path, target_w, target_h, num_points)
        logos.append(pts)
        
    if not logos:
        return None, None
        
    # Stabilize the cycle
    L3 = logos[-1]
    cycle = [L3]
    for i in range(len(logos) - 1):
        nxt = optimal_transport(cycle[-1], logos[i])
        cycle.append(nxt)
        
    # L3 -> L1 -> L2... Let's do it cleanly for exactly 3 logos
    # logos[0]=L1, logos[1]=L2, logos[2]=L3
    L1 = optimal_transport(logos[2], logos[0])
    L2 = optimal_transport(L1, logos[1])
    L3 = optimal_transport(L2, logos[2])
    
    # Now sample random starting points from portrait
    if len(portrait_pts) >= num_points:
        idx = np.random.choice(len(portrait_pts), num_points, replace=False)
    else:
        idx = np.random.choice(len(portrait_pts), num_points, replace=True)
    P0_initial = portrait_pts[idx]
    
    P0 = optimal_transport(L3, P0_initial)
    
    # Routes is a list of arrays: [P0, L1, L2, L3]
    return [P0, L1, L2, L3]


def compute_drift_bands(portrait_pts, logo1_centroid, num_bands=94):
    """
    Group portrait dots into bands with organic boundaries, and compute their drift vectors.
    """
    # Add noise to positions
    noisy_pts = portrait_pts + np.random.normal(0, 4, portrait_pts.shape)
    
    # Find bounding box
    min_x, max_x = np.min(noisy_pts[:, 0]), np.max(noisy_pts[:, 0])
    min_y, max_y = np.min(noisy_pts[:, 1]), np.max(noisy_pts[:, 1])
    
    # Quantize into a grid to form roughly `num_bands` groups
    grid_cols = int(math.sqrt(num_bands * (max_x - min_x) / (max_y - min_y)))
    grid_rows = int(num_bands / grid_cols)
    
    groups = {}
    cx, cy = logo1_centroid
    
    for i, (x, y) in enumerate(portrait_pts):
        nx, ny = noisy_pts[i]
        
        col = int((nx - min_x) / (max_x - min_x + 1e-5) * grid_cols)
        row = int((ny - min_y) / (max_y - min_y + 1e-5) * grid_rows)
        
        col = max(0, min(grid_cols - 1, col))
        row = max(0, min(grid_rows - 1, row))
        
        key = (row, col)
        if key not in groups:
            groups[key] = []
        groups[key].append((x, y))
        
    bands = []
    for pts in groups.values():
        pts = np.array(pts)
        # Average actual drift
        # Drift vector points towards the centroid
        avg_x, avg_y = np.mean(pts, axis=0)
        dx = (cx - avg_x) * 0.42
        dy = (cy - avg_y) * 0.42
        bands.append({'pts': pts, 'dx': dx, 'dy': dy})
        
    return bands

