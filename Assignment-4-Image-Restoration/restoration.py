"""
Name: Jatin Bisht
Roll No: 2301010360
Course: Image Processing & Computer Vision
Unit: Noise Modeling & Restoration (Modified Version)
Assignment Title: Advanced Image Restoration using Python
Date:
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== Advanced Image Restoration System for Surveillance ===")

# Create output folder
output_folder = "jatin_restoration_outputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# -------------------------------
# Task 1: Load Image
# -------------------------------
image_path = "jatin_input.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found!")
    exit()

img = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite(f"{output_folder}/original.jpg", gray)

# -------------------------------
# Task 2: Noise Modeling
# -------------------------------

# Gaussian Noise
gaussian = gray + np.random.normal(0, 25, gray.shape)
gaussian = np.clip(gaussian, 0, 255).astype(np.uint8)

# Salt & Pepper Noise
sp = gray.copy()
prob = 0.02
rnd = np.random.rand(*gray.shape)
sp[rnd < prob] = 0
sp[rnd > 1 - prob] = 255

cv2.imwrite(f"{output_folder}/gaussian_noise.jpg", gaussian)
cv2.imwrite(f"{output_folder}/sp_noise.jpg", sp)

# -------------------------------
# Task 3: Restoration
# -------------------------------

def apply_filters(image):
    mean = cv2.blur(image, (5,5))
    median = cv2.medianBlur(image, 5)
    gaussian_f = cv2.GaussianBlur(image, (5,5), 0)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)  # NEW FILTER
    return mean, median, gaussian_f, bilateral

g_mean, g_median, g_gauss, g_bilateral = apply_filters(gaussian)
sp_mean, sp_median, sp_gauss, sp_bilateral = apply_filters(sp)

# Save outputs
cv2.imwrite(f"{output_folder}/g_mean.jpg", g_mean)
cv2.imwrite(f"{output_folder}/g_median.jpg", g_median)
cv2.imwrite(f"{output_folder}/g_gauss.jpg", g_gauss)
cv2.imwrite(f"{output_folder}/g_bilateral.jpg", g_bilateral)

cv2.imwrite(f"{output_folder}/sp_mean.jpg", sp_mean)
cv2.imwrite(f"{output_folder}/sp_median.jpg", sp_median)
cv2.imwrite(f"{output_folder}/sp_gauss.jpg", sp_gauss)
cv2.imwrite(f"{output_folder}/sp_bilateral.jpg", sp_bilateral)

# -------------------------------
# Extra Feature: Sharpening
# -------------------------------
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpened = cv2.filter2D(g_gauss, -1, kernel)
cv2.imwrite(f"{output_folder}/sharpened.jpg", sharpened)

# -------------------------------
# Task 4: Metrics
# -------------------------------

def mse(original, restored):
    return np.mean((original - restored) ** 2)

def psnr(original, restored):
    m = mse(original, restored)
    if m == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(m))

print("\n=== Performance (Gaussian Noise) ===")
print("Mean Filter       -> MSE:", mse(gray, g_mean), " PSNR:", psnr(gray, g_mean))
print("Median Filter     -> MSE:", mse(gray, g_median), " PSNR:", psnr(gray, g_median))
print("Gaussian Filter   -> MSE:", mse(gray, g_gauss), " PSNR:", psnr(gray, g_gauss))
print("Bilateral Filter  -> MSE:", mse(gray, g_bilateral), " PSNR:", psnr(gray, g_bilateral))

print("\n=== Performance (Salt & Pepper Noise) ===")
print("Mean Filter       -> MSE:", mse(gray, sp_mean), " PSNR:", psnr(gray, sp_mean))
print("Median Filter     -> MSE:", mse(gray, sp_median), " PSNR:", psnr(gray, sp_median))
print("Gaussian Filter   -> MSE:", mse(gray, sp_gauss), " PSNR:", psnr(gray, sp_gauss))
print("Bilateral Filter  -> MSE:", mse(gray, sp_bilateral), " PSNR:", psnr(gray, sp_bilateral))

# -------------------------------
# Task 5: Display
# -------------------------------

titles = [
    "Original", "Gaussian Noise", "SP Noise",
    "G-Mean", "G-Median", "G-Gaussian",
    "G-Bilateral", "Sharpened"
]

images = [
    gray, gaussian, sp,
    g_mean, g_median, g_gauss,
    g_bilateral, sharpened
]

plt.figure(figsize=(12,8))

for i in range(len(images)):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig(f"{output_folder}/comparison.png")
plt.show()

# -------------------------------
# Analysis
# -------------------------------

print("\n=== Analysis ===")
print("Gaussian Noise: Gaussian and Bilateral filters perform well.")
print("Salt & Pepper Noise: Median filter performs best.")
print("Bilateral filter preserves edges better than Gaussian filter.")
print("Sharpening enhances final image clarity.")