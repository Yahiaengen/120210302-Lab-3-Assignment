
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def load_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

def extract_green_regions(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = img_lab[:, :, 1].astype(np.float64)
    a_channel = (a_channel - np.mean(a_channel)) / np.std(a_channel)
    flat_data = a_channel.reshape(-1, 1)
    cluster_model = GreenDigitKMeans(clusters=2, strategy='++', attempts=10, seed=42)
    cluster_model.fit(flat_data)
    label_map = cluster_model.predict(flat_data).reshape(a_channel.shape)
    green_cluster = np.argmax(cluster_model.centroids)
    green_mask = np.zeros_like(img_rgb)
    green_mask[label_map == green_cluster] = [0, 255, 0]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(green_mask)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

class GreenDigitKMeans:
    def __init__(self, clusters=2, strategy='++', attempts=5, max_steps=300, tolerance=1e-4, seed=None):
        self.clusters = clusters
        self.strategy = strategy
        self.attempts = attempts
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.rng = np.random.RandomState(seed)
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.iterations = 0

    def _init_centroids(self, data):
        n, d = data.shape
        centers = np.empty((self.clusters, d), dtype=data.dtype)
        centers[0] = data[self.rng.choice(n)]
        dist_sq = np.full(n, np.inf)
        for i in range(1, self.clusters):
            dist_sq = np.minimum(dist_sq, np.sum((data - centers[i-1])**2, axis=1))
            prob = dist_sq / dist_sq.sum()
            centers[i] = data[self.rng.choice(n, p=prob)]
        return centers

    def _run_lloyd(self, data, centers):
        for i in range(self.max_steps):
            dists = np.sum((data[:, None] - centers)**2, axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = np.empty_like(centers)
            for j in range(self.clusters):
                group = data[labels == j]
                new_centers[j] = group.mean(axis=0) if len(group) > 0 else data[self.rng.choice(len(data))]
            shift = np.sum((centers - new_centers)**2)
            if shift <= self.tolerance:
                break
            centers = new_centers
        final_dists = np.sum((data[:, None] - centers)**2, axis=2)
        final_labels = np.argmin(final_dists, axis=1)
        inertia = np.sum(final_dists[np.arange(len(final_dists)), final_labels])
        return centers, final_labels, inertia, i + 1

    def fit(self, data):
        best_inertia = np.inf
        for _ in range(self.attempts):
            init = self._init_centroids(data) if self.strategy == '++' else data[self.rng.choice(len(data), self.clusters)]
            centers, labels, inertia, iters = self._run_lloyd(data, init)
            if inertia < best_inertia:
                self.centroids = centers
                self.labels = labels
                self.inertia = inertia
                self.iterations = iters
                best_inertia = inertia
        return self

    def predict(self, data):
        dists = np.sum((data[:, None] - self.centroids)**2, axis=2)
        return np.argmin(dists, axis=1)

if __name__ == "__main__":
    selected_image = load_image()
    if selected_image:
        print(f"Processing: {selected_image}")
        extract_green_regions(selected_image)
    else:
        print("No image selected.")
