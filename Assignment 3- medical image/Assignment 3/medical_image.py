"""
Name: Jatin Bisht
Roll No: 2301010360
Course: Image Processing & Computer Vision
Unit: 1
Assignment Title: Medical Image Compression & Segmentation System
Date:
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== Smart Medical Image Compression & Segmentation System ===")

# -------------------------------
# Create output folder
# -------------------------------
output_folder = "jatin_medical_outputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# -------------------------------
# Load Image
# -------------------------------
print("\n[INFO] Loading medical image...")

image_path = "jatin_medical.jpg"
img = cv2.imread(image_path, 0)

if img is None:
    print("Error: Image not found!")
    exit()

img = cv2.resize(img, (512, 512))
cv2.imwrite(f"{output_folder}/original.jpg", img)

# -------------------------------
# Task 1: RLE Compression
# -------------------------------
print("\n[INFO] Performing RLE Compression...")

def rle_encode(image):
    pixels = image.flatten()
    encoding = []
    prev = pixels[0]
    count = 1

    for pixel in pixels[1:]:
        if pixel == prev:
            count += 1
        else:
            encoding.append((prev, count))
            prev = pixel
            count = 1

    encoding.append((prev, count))
    return encoding

rle = rle_encode(img)

original_size = img.size
compressed_size = len(rle) * 2

compression_ratio = original_size / compressed_size
savings = (1 - compressed_size / original_size) * 100

print("\n=== Compression Results ===")
print("Original Size:", original_size)
print("Compressed Size:", compressed_size)
print("Compression Ratio:", round(compression_ratio, 2))
print("Storage Savings (%):", round(savings, 2))

# Save partial RLE
with open(f"{output_folder}/rle.txt", "w") as f:
    f.write(str(rle[:100]))

# -------------------------------
# Task 2: Segmentation
# -------------------------------
print("\n[INFO] Performing Image Segmentation...")

# Global Threshold
_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu Threshold
_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite(f"{output_folder}/global_thresh.jpg", global_thresh)
cv2.imwrite(f"{output_folder}/otsu_thresh.jpg", otsu_thresh)

# -------------------------------
# Task 3: Morphological Processing
# -------------------------------
print("\n[INFO] Applying Morphological Operations...")

kernel = np.ones((3,3), np.uint8)

dilation = cv2.dilate(otsu_thresh, kernel, iterations=1)
erosion = cv2.erode(otsu_thresh, kernel, iterations=1)

cv2.imwrite(f"{output_folder}/dilation.jpg", dilation)
cv2.imwrite(f"{output_folder}/erosion.jpg", erosion)

# -------------------------------
# Display Results
# -------------------------------
print("\n[INFO] Displaying Results...")

titles = [
    "Original",
    "Global Threshold",
    "Otsu Threshold",
    "Dilation",
    "Erosion"
]

images = [
    img,
    global_thresh,
    otsu_thresh,
    dilation,
    erosion
]

plt.figure(figsize=(12,7))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig(f"{output_folder}/comparison.png")
plt.show()

# -------------------------------
# Analysis (VERY IMPORTANT)
# -------------------------------
print("\n=== Analysis & Clinical Interpretation ===")

print("1. RLE compression reduces redundancy in medical images without losing information.")
print("2. High compression ratio indicates efficient storage, useful in hospitals.")
print("3. Global thresholding segments basic regions but may fail under varying intensities.")
print("4. Otsu’s method automatically selects optimal threshold and provides better segmentation.")
print("5. Dilation helps in filling gaps in segmented regions (useful for highlighting organs).")
print("6. Erosion removes small noise and refines boundaries.")
print("7. These techniques help identify regions of interest like bones, tissues, or abnormalities.")
print("8. This system simulates real-world medical imaging pipelines used in diagnostics.")