import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys


# ==========================================
# PART A: "FROM SCRATCH" IMPLEMENTATION
# ==========================================

class ProjectScratch:
    def __init__(self, k_objects=1):
        self.k = k_objects
        self.tracks = [[] for _ in range(k_objects)]  # Store (t, x, y) for each object

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def k_means(self, pixels, k, max_iters=10):
        # 1. Initialize centroids randomly from the data points
        if len(pixels) < k: return []
        centroids = pixels[np.random.choice(len(pixels), k, replace=False)]

        for _ in range(max_iters):
            # 2. Assign pixels to nearest centroid
            # Calculate distances: Shape (N_pixels, K_centroids)
            distances = np.linalg.norm(pixels[:, None] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = []
            for i in range(k):
                # 3. Update centroids
                cluster_points = pixels[labels == i]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    new_centroids.append(centroids[i])  # Keep old if empty

            new_centroids = np.array(new_centroids)
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return centroids

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Grayscale for scratch logic
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Thresholding (Manual logic: check if pixel > 10)
            # Using numpy for speed, but logic is basic element-wise comparison
            y_coords, x_coords = np.where(gray > 20)

            if len(x_coords) > 0:
                pixel_data = np.column_stack((x_coords, y_coords))

                # Perform Clustering
                centroids = self.k_means(pixel_data, self.k)

                # Tracking: Associate new centroids with previous tracks
                # Sort centroids by x-coordinate to maintain consistency (simple heuristic)
                # specific to the provided video where objects don't cross over much
                centroids = sorted(centroids, key=lambda p: p[0])

                for i in range(min(len(centroids), self.k)):
                    self.tracks[i].append((frame_idx / fps, centroids[i][0], centroids[i][1]))

            frame_idx += 1
        cap.release()
        return self.tracks

    def calculate_derivatives(self):
        results = []
        for track in self.tracks:
            data = np.array(track)
            if len(data) < 5: continue

            t = data[:, 0]
            x = data[:, 1]
            y = data[:, 2]

            # Smoothing (Moving Average)
            window = 5
            x_smooth = np.convolve(x, np.ones(window) / window, mode='valid')
            y_smooth = np.convolve(y, np.ones(window) / window, mode='valid')
            t_valid = t[window - 1:]

            dt = t[1] - t[0]  # Assuming constant fps

            # Finite Differences
            def get_diff(arr, dt):
                return np.diff(arr) / dt

            vx = get_diff(x_smooth, dt)
            ax = get_diff(vx, dt)
            jx = get_diff(ax, dt)  # Jerk
            sx = get_diff(jx, dt)  # Jounce

            # Pad arrays for plotting consistency
            results.append({
                't': t_valid[:-4],  # Truncate time to match shortest derivative
                'x': x_smooth[:-4],
                'v': vx[:-3],
                'a': ax[:-2],
                'j': jx[:-1],
                's': sx
            })
        return results


# ==========================================
# PART B: "LIBRARY" IMPLEMENTATION
# ==========================================

class ProjectLibrary:
    def __init__(self):
        self.trajectories = {}  # ID -> lists of (t, x, y)
        self.next_id = 0

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 1. Preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

            # 2. Contours (Blob Detection)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_centroids = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    current_centroids.append((cX, cY))

            # 3. Tracking (Nearest Neighbor)
            # Simple association for demonstration
            if frame_idx == 0:
                for c in current_centroids:
                    self.trajectories[self.next_id] = [(0, c[0], c[1])]
                    self.next_id += 1
            else:
                # Match current centroids to existing trajectories
                for t_id, points in self.trajectories.items():
                    last_pos = points[-1]
                    # Find closest current centroid
                    best_dist = float('inf')
                    best_match = None

                    for c in current_centroids:
                        dist = np.linalg.norm(np.array(c) - np.array((last_pos[1], last_pos[2])))
                        if dist < best_dist:
                            best_dist = dist
                            best_match = c

                    if best_match and best_dist < 50:  # Threshold for tracking
                        self.trajectories[t_id].append((frame_idx / fps, best_match[0], best_match[1]))

            frame_idx += 1
        cap.release()
        return self.trajectories

    def calculate_derivatives(self):
        results = []
        for t_id, points in self.trajectories.items():
            data = np.array(points)
            if len(data) < 10: continue

            t = data[:, 0]
            x = data[:, 1]

            # Smoothing (Savitzky-Golay) - Superior to moving average
            window_length = 15  # Must be odd
            poly_order = 3
            if len(x) > window_length:
                x_smooth = savgol_filter(x, window_length, poly_order)
            else:
                x_smooth = x

            # Numpy Gradient (Central Difference - more accurate)
            v = np.gradient(x_smooth, t)
            a = np.gradient(v, t)
            j = np.gradient(a, t)
            s = np.gradient(j, t)

            results.append({
                't': t,
                'x': x_smooth,
                'v': v,
                'a': a,
                'j': j,
                's': s
            })
        return results


# ==========================================
# VISUALIZATION & RUNNER
# ==========================================

def plot_results(results, title):
    fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(title)

    labels = ['Position (px)', 'Velocity (px/s)', 'Acceleration (px/s²)', 'Jerk (px/s³)', 'Jounce (px/s⁴)']
    keys = ['x', 'v', 'a', 'j', 's']

    for obj_res in results:
        t = obj_res['t']
        for i, key in enumerate(keys):
            axs[i].plot(t, obj_res[key], label='Object')
            axs[i].set_ylabel(labels[i])
            axs[i].grid(True)

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Set this to the path of your video files
    video_single = r"C:\Users\lenovo\Downloads\video_single.mp4"
    video_multi = r"C:\Users\lenovo\Downloads\video_multi.mp4"

    print("Running Scratch Implementation on Single Object Video...")
    scratch_solver = ProjectScratch(k_objects=1)
    scratch_solver.process_video(video_single)
    res_scratch = scratch_solver.calculate_derivatives()
    plot_results(res_scratch, "SCRATCH METHOD: Single Object Kinematics")

    print("Running Library Implementation on Multi Object Video...")
    lib_solver = ProjectLibrary()
    lib_solver.process_video(video_multi)
    res_lib = lib_solver.calculate_derivatives()
    plot_results(res_lib, "LIBRARY METHOD: Multi Object Kinematics")

    print("Done.")