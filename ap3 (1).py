import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import cv2
from PIL import Image, ImageDraw
import os
import sys

# ==========================================
# 1. PARAMETERS
# ==========================================
params = {
    'N': 1200,  # 1200 Drones for high clarity
    'dt': 0.05,  # Time step
    'm': 1.0,  # Mass
    'kp': 2.5,  # Stiffer attraction
    'kd': 1.5,  # Damping
    'k_rep': 0.8,  # Lower repulsion
    'r_safe': 0.8,  # Smaller safety radius
    'v_max': 25.0,  # Faster max speed
    'width': 100,  # Workspace width
    'height': 100,  # Workspace height
    'frames_static': 150,
    'frames_dynamic': 200
}


# ==========================================
# 2. CORE FUNCTIONS
# ==========================================

def image_to_points(image_path, num_points, target_width, target_height):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[CRITICAL ERROR] '{image_path}' not found.")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"[CRITICAL ERROR] Could not read '{image_path}'.")

    # Process at 4x resolution
    process_w, process_h = target_width * 4, target_height * 4
    img = cv2.resize(img, (process_w, process_h))

    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(img, 100, 200)
    y_idxs, x_idxs = np.where(edges > 0)

    if len(x_idxs) == 0:
        raise ValueError(f"[ERROR] No edges detected in '{image_path}'.")

    # Normalize back to 0-100 range
    points_x = x_idxs / 4.0
    points_y = (process_h - y_idxs) / 4.0
    points = np.column_stack((points_x, points_y))

    if len(points) < num_points:
        print(f"[WARNING] Image has fewer pixels ({len(points)}) than drones ({num_points}). Duplicating points.")
        extra_indices = np.random.choice(len(points), num_points - len(points))
        extra_points = points[extra_indices]
        points = np.vstack((points, extra_points))

    indices = np.linspace(0, len(points) - 1, num_points).astype(int)
    return points[indices]


def assign_targets_simple(drones_pos, targets):
    p_indices = np.argsort(drones_pos[:, 0])
    t_indices = np.argsort(targets[:, 0])
    final_targets = np.zeros_like(targets)
    final_targets[p_indices] = targets[t_indices]
    return final_targets


def compute_forces(X, V, T, p):
    F_att = p['kp'] * (T - X)
    F_damp = -p['kd'] * V

    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)

    mask = dist < p['r_safe']
    with np.errstate(divide='ignore'):
        factor = np.where(mask, p['k_rep'] / (dist ** 3), 0.0)

    F_rep = np.sum(diff * factor[:, :, np.newaxis], axis=1)
    return (F_att + F_damp + F_rep) / p['m']


def explicit_euler_step(X, V, T, p):
    A = compute_forces(X, V, T, p)
    V_new = V + p['dt'] * A

    speeds = np.linalg.norm(V_new, axis=1, keepdims=True)
    scale = np.minimum(1.0, p['v_max'] / (speeds + 1e-8))
    V_sat = V_new * scale

    X_new = X + p['dt'] * V_sat
    return X_new, V_sat


# ==========================================
# 3. MAIN SIMULATION
# ==========================================

def run_simulation():
    print(f"Initializing Swarm with {params['N']} drones...")
    X = np.random.rand(params['N'], 2) * 100
    V = np.zeros((params['N'], 2))
    history = []

    # --- TASK 1: name.png ---
    print(f"--- Task 1: name.png ---")
    try:
        target_1 = image_to_points("name.png", params['N'], 100, 100)
    except Exception as e:
        print(e)
        sys.exit(1)

    T = assign_targets_simple(X, target_1)
    for _ in range(params['frames_static']):
        X, V = explicit_euler_step(X, V, T, params)
        history.append(X)

    # --- TASK 2: greeting.png ---
    print(f"--- Task 2: greeting.png ---")
    try:
        # UPDATED FILENAME HERE
        target_2 = image_to_points("greeting.png", params['N'], 100, 100)
    except Exception as e:
        print(e)
        sys.exit(1)

    T = assign_targets_simple(X, target_2)
    for _ in range(params['frames_static']):
        X, V = explicit_euler_step(X, V, T, params)
        history.append(X)

    # --- TASK 3: Tracking ---
    print(f"--- Task 3: Tracking ---")
    center_x = 20
    for t in range(params['frames_dynamic']):
        img = Image.new('L', (400, 400), 0)
        draw = ImageDraw.Draw(img)
        center_x += 2.0
        draw.ellipse([center_x - 40, 160, center_x + 40, 240], outline=255, width=5)

        edges = cv2.Canny(np.array(img), 50, 150)
        y_idxs, x_idxs = np.where(edges > 0)

        if len(x_idxs) > 0:
            raw = np.column_stack((x_idxs / 4.0, (400 - y_idxs) / 4.0))
            if len(raw) >= params['N']:
                idx = np.linspace(0, len(raw) - 1, params['N']).astype(int)
                T = assign_targets_simple(X, raw[idx])
            else:
                idx = np.random.choice(len(raw), params['N'])
                T = assign_targets_simple(X, raw[idx])

        X, V = explicit_euler_step(X, V, T, params)
        history.append(X)

    return history


if __name__ == "__main__":
    hist = run_simulation()

    print("Rendering High-Density Animation...")
    fig, ax = plt.subplots(figsize=(8, 8))
    scat = ax.scatter([], [], s=2, c='cyan', alpha=0.7)

    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    t1_end = params['frames_static']
    t2_end = t1_end + params['frames_static']


    def update(frame):
        scat.set_offsets(hist[frame])
        return scat,


    anim = FuncAnimation(fig, update, frames=len(hist), interval=20, blit=True)

    output_file = 'drone_show_hd.mp4'
    print(f"Saving high-res video to {output_file}...")
    try:
        writer = FFMpegWriter(fps=30, bitrate=3000)
        anim.save(output_file, writer=writer)
        print(f"[SUCCESS] Saved {output_file}")
    except Exception as e:
        print(f"[ERROR] FFmpeg missing? {e}")
        plt.show()