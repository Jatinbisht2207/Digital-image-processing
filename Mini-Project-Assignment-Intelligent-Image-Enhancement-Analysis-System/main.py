"""
Name: Jatin Bisht
Roll No: YOUR_ROLL_NO
Course: Image Processing & Computer Vision
Assignment Title: Smart Image Enhancement & Analysis Tool
Date:
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim

print("=== Smart Image Enhancement & Analysis Tool ===")
output_folder = "jatin_outputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Task 1: Image Acquisition
# ------------------------------
image_path = "jatin_input.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found!")
    exit()

img = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite(f"{output_folder}/original.jpg", img)
cv2.imwrite(f"{output_folder}/grayscale.jpg", gray)

# -------------------------------
# Task 2: Noise + Restoration
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

# Filters
def apply_filters(image):
    mean = cv2.blur(image, (5,5))
    median = cv2.medianBlur(image, 5)
    gauss = cv2.GaussianBlur(image, (5,5), 0)
    return mean, median, gauss

g_mean, g_median, g_gauss = apply_filters(gaussian)

# Choose best restored
restored = g_gauss

# -------------------------------
# Task 3: Enhancement 
# -------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(restored)

# Custom Feature: Brightness Adjustment
bright = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=30)

cv2.imwrite(f"{output_folder}/gaussian_noise.jpg", gaussian)
cv2.imwrite(f"{output_folder}/sp_noise.jpg", sp)
cv2.imwrite(f"{output_folder}/restored.jpg", restored)
cv2.imwrite(f"{output_folder}/enhanced.jpg", enhanced)
cv2.imwrite(f"{output_folder}/brightened.jpg", bright)

# -------------------------------
# Task 4: Segmentation
# -------------------------------
_, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(otsu, kernel, iterations=1)

cv2.imwrite(f"{output_folder}/segmented.jpg", otsu)
cv2.imwrite(f"{output_folder}/morphology.jpg", dilation)

# -------------------------------
# Task 5: Feature Extraction
# -------------------------------

# Edges
canny = cv2.Canny(enhanced, 100, 200)

# Contours
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()

for cnt in contours:
    if cv2.contourArea(cnt) > 500:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x,y), (x+w,y+h), (0,255,0), 2)

# ORB Features
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(enhanced, None)
feature_img = cv2.drawKeypoints(img, kp, None, color=(0,255,0))

cv2.imwrite(f"{output_folder}/edges.jpg", canny)
cv2.imwrite(f"{output_folder}/contours.jpg", contour_img)
cv2.imwrite(f"{output_folder}/features.jpg", feature_img)

# -------------------------------
# Task 6: Performance Metrics
# -------------------------------
def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(m))

print("\n=== Performance ===")
print("MSE:", mse(gray, enhanced))
print("PSNR:", psnr(gray, enhanced))
print("SSIM:", ssim(gray, enhanced))

# -------------------------------
# Task 7: Visualization
# -------------------------------
titles = ["Original", "Noisy", "Restored", "Enhanced", "Segmented", "Features"]

images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    gaussian,
    restored,
    enhanced,
    otsu,
    cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB)
]

plt.figure(figsize=(10,6))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig(f"{output_folder}/comparison.png")
plt.show()

# -------------------------------
# Conclusion
# -------------------------------
print("\n=== Conclusion ===")
print("This system enhances image quality using CLAHE, applies segmentation, and extracts features effectively.")