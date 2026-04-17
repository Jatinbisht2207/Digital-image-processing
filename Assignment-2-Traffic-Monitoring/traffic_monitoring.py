"""
Name: Jatin Bisht
Roll No: 2301010360
Course: Image Processing & Computer Vision
Unit: 1
Assignment Title: Feature-Based Traffic Monitoring System (Modified)
Date:
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== Smart Traffic Monitoring & Feature Extraction System ===")

# -------------------------------
# Create output folder
# -------------------------------
output_folder = "jatin_traffic_outputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# -------------------------------
# Load Image
# -------------------------------
print("\n[INFO] Loading traffic image...")
image_path = "jatin_input.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found!")
    exit()

img = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite(f"{output_folder}/original.jpg", img)

# -------------------------------
# Task 1: Edge Detection
# -------------------------------
print("\n[INFO] Performing Edge Detection...")

# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(np.clip(sobel, 0, 255))

# Canny
canny = cv2.Canny(gray, 100, 200)

cv2.imwrite(f"{output_folder}/sobel.jpg", sobel)
cv2.imwrite(f"{output_folder}/canny.jpg", canny)

# -------------------------------
# Task 2: Object Representation
# -------------------------------
print("\n[INFO] Detecting objects using contours...")

contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = img.copy()
vehicle_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area > 800:  # improved filtering
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x,y), (x+w, y+h), (0,255,0), 2)

        vehicle_count += 1
        print(f"Vehicle {vehicle_count} -> Area: {area:.2f}, Perimeter: {perimeter:.2f}")

cv2.imwrite(f"{output_folder}/contours.jpg", contour_img)

print("\nTotal Detected Objects (Vehicles Approx):", vehicle_count)

# -------------------------------
# Task 3: Feature Extraction (ORB)
# -------------------------------
print("\n[INFO] Extracting ORB features...")

orb = cv2.ORB_create(nfeatures=500)
keypoints, descriptors = orb.detectAndCompute(gray, None)

feature_img = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0))

cv2.imwrite(f"{output_folder}/orb_features.jpg", feature_img)

print("Total Keypoints Detected:", len(keypoints))

# -------------------------------
# Display Results
# -------------------------------
print("\n[INFO] Displaying Results...")

titles = [
    "Original", "Sobel Edge", "Canny Edge",
    "Contours (Objects)", "ORB Features"
]

images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    sobel, canny,
    cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB)
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
print("\n=== Comparative Analysis ===")
print("1. Canny edge detector provides sharper and well-defined edges compared to Sobel.")
print("2. Sobel highlights gradient intensity but is less precise for object boundaries.")
print("3. Contours help identify vehicles and objects on the road.")
print("4. Area and perimeter provide geometric representation of detected objects.")
print("5. ORB detects keypoints useful for tracking vehicles across frames.")
print("6. This system simulates real-world traffic monitoring applications like smart cities.")