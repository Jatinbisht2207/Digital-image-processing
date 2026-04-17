# Feature-Based Traffic Monitoring System

## Course

Image Processing & Computer Vision

## Assignment

Object Representation and Feature Extraction for Traffic Images

## Student Details

* Name: Jatin Bisht
* Roll No: 2301010360
* Unit: 1

---

## Project Overview

Traffic monitoring systems analyze road scenes to detect vehicles, pedestrians, and lanes.
This project implements a feature-based approach to understand and process traffic images using classical computer vision techniques.

The system focuses on detecting edges, identifying objects, and extracting key features that can be used in real-world intelligent transportation systems.

---

## Objectives

* Apply edge detection techniques such as Sobel and Canny
* Detect and represent objects using contours and bounding boxes
* Compute object properties such as area and perimeter
* Extract key features using ORB (Oriented FAST and Rotated BRIEF)
* Compare different edge detection and feature extraction techniques
* Understand how these techniques are used in traffic monitoring systems

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib

---

## Methodology

1. Load and preprocess the traffic image
2. Perform edge detection using Sobel and Canny operators
3. Detect contours and draw bounding boxes around objects
4. Compute area and perimeter of detected objects
5. Extract features using ORB and visualize keypoints
6. Compare outputs and analyze performance

---

## Output

The system generates the following outputs:

* Original image
* Sobel edge detection result
* Canny edge detection result
* Contour detection with bounding boxes
* ORB feature extraction visualization
* Comparison image showing all results

All outputs are stored in the output folder.

---

## Analysis

* Canny edge detection provides sharper and more accurate edges compared to Sobel
* Sobel highlights intensity gradients but lacks precise edge localization
* Contours help in identifying object boundaries such as vehicles
* Area and perimeter provide useful geometric representation of objects
* ORB features are useful for tracking and recognition tasks in traffic systems

---

## Conclusion

This project demonstrates how classical image processing techniques can be combined to build a basic traffic monitoring system.
The approach provides a foundation for advanced applications such as vehicle detection, tracking, and smart traffic management.

---
