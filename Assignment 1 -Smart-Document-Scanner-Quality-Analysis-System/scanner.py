"""
Name: Jatin Bisht
Roll No: 2301010360
Course: Image Processing & Computer Vision
Assignment Title: Smart Document Scanner & Quality Analysis System
Date:
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== Smart Document Scanner & Quality Analysis System ===")
print("This system simulates real-world document digitization and analyzes quality degradation.\n")

# -------------------------------
# Create Output Folder
# -------------------------------
output_folder = "jatin_scanner_outputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# -------------------------------
# Task 2: Image Acquisition
# -------------------------------
print("[INFO] Loading document image...")

image_path = "jatin_document.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found!")
    exit()

# Resize to standard size
img = cv2.resize(img, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite(f"{output_folder}/original.jpg", img)
cv2.imwrite(f"{output_folder}/grayscale.jpg", gray)

# -------------------------------
# Task 3: Sampling (Resolution)
# -------------------------------
print("[INFO] Performing Sampling Analysis...")

def sample_image(image, size):
    small = cv2.resize(image, size)
    upscaled = cv2.resize(small, (512, 512))
    return upscaled

high_res = gray.copy()
med_res = sample_image(gray, (256, 256))
low_res = sample_image(gray, (128, 128))

cv2.imwrite(f"{output_folder}/sample_512.jpg", high_res)
cv2.imwrite(f"{output_folder}/sample_256.jpg", med_res)
cv2.imwrite(f"{output_folder}/sample_128.jpg", low_res)

# -------------------------------
# Task 4: Quantization
# -------------------------------
print("[INFO] Performing Quantization Analysis...")

def quantize(image, levels):
    factor = 256 // levels
    quantized = (image // factor) * factor
    return quantized

q_8bit = quantize(gray, 256)
q_4bit = quantize(gray, 16)
q_2bit = quantize(gray, 4)

cv2.imwrite(f"{output_folder}/quant_8bit.jpg", q_8bit)
cv2.imwrite(f"{output_folder}/quant_4bit.jpg", q_4bit)
cv2.imwrite(f"{output_folder}/quant_2bit.jpg", q_2bit)

# -------------------------------
# Display Comparison
# -------------------------------
print("[INFO] Displaying Results...")

titles = [
    "Original", "Grayscale",
    "512x512", "256x256", "128x128",
    "8-bit", "4-bit", "2-bit"
]

images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gray,
    high_res, med_res, low_res,
    q_8bit, q_4bit, q_2bit
]

plt.figure(figsize=(12, 8))

for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig(f"{output_folder}/comparison.png")
plt.show()

# -------------------------------
# Task 5: Observations & Analysis
# -------------------------------
print("\n=== Detailed Analysis ===")

print("\n1. Sampling Effects (Resolution):")
print("- High Resolution (512x512): Clear text, sharp edges, best readability")
print("- Medium Resolution (256x256): Slight blur, readable but reduced clarity")
print("- Low Resolution (128x128): Significant loss of text details and edges")

print("\n2. Quantization Effects (Gray Levels):")
print("- 8-bit (256 levels): No visible loss, best quality")
print("- 4-bit (16 levels): Minor banding, acceptable for reading")
print("- 2-bit (4 levels): Severe loss of detail, poor readability")

print("\n3. OCR Suitability:")
print("- Best: High resolution + 8-bit (ideal for OCR systems)")
print("- Acceptable: Medium resolution + 4-bit")
print("- Poor: Low resolution + 2-bit (OCR may fail)")

print("\n4. Real-World Insight:")
print("This analysis shows how improper scanning or compression can degrade document quality.")
print("High resolution and higher bit-depth are critical for accurate digitization in banks, offices, and legal systems.")