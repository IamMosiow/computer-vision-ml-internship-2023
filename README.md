# Computer Vision and Machine Learning Internship Projects (2023)

**Author:** Seyyed Mostafa Mousavi  
**Languages:** Python, C++  
**Libraries Used:** OpenCV, NumPy, TensorFlow/Keras, PyTorch (some projects)  

---

## 📘 Overview

This repository contains a collection of **computer vision** and **machine learning** projects developed during my internship in 2023.  
They cover a range of foundational and intermediate topics such as image classification, motion detection, object tracking, background subtraction, and face recognition.

These projects were part of my learning and experimentation journey, and most of them were implemented from scratch or built on standard machine learning datasets like **MNIST** and **custom face datasets**.

---

## 🧩 Projects Included

| Project Folder | Description |
|----------------|-------------|
| **basic_motion_detection/** | Detects motion in a video stream using frame differencing and contour analysis. |
| **detect_and_classify_digits/** | Identifies handwritten digits using image processing and a trained digit recognition model. |
| **faces_dnn/** | Face detection using deep neural networks (DNN module of OpenCV). |
| **generating_face_data/** | Captures and stores facial data for building a recognition dataset. |
| **handwritten_digits_classification/** | Classic digit classification pipeline using image preprocessing and ML models. |
| **loading_and_training_face_recognition/** | Loads datasets, trains facial recognition models, and evaluates accuracy. |
| **logo_matching/** | Uses template matching and keypoint detection (ORB/SIFT) to find a logo in another image. |
| **mnist_gui_recognition/** | A simple graphical interface for MNIST digit recognition using OpenCV or Tkinter. |
| **mnist_with_mlps/** | MNIST handwritten digit recognition using Multi-Layer Perceptrons (MLPs). |
| **object_detection/** | Basic object detection using pre-trained DNNs or contour-based techniques. |
| **background_estimation/** | Performs simple background subtraction using running average or median filtering. |
| **srcs/** | Shared helper functions and utilities used across multiple projects. |
| **trained_model/** | Pretrained models saved from training scripts. |
| **using_mnist_trained_model/** | Demonstrates inference on new images using a previously trained MNIST model. |

---

## ⚙️ Dependencies

Make sure you have the following installed:

```bash
pip install opencv-python numpy tensorflow torch matplotlib

For C++ projects, you’ll need:

OpenCV (built with videoio & highgui)

A standard C++17 compiler (e.g., GCC, MSVC)

## ▶️ How to Run

Each folder contains a standalone project.
To run a Python project:

cd project_folder_name
python main.py


For C++ projects:

cd project_folder_name
g++ main.cpp -o main `pkg-config --cflags --libs opencv4`
./main

## 📊 Highlights

Implemented both classical and deep-learning-based methods for vision tasks.

Built end-to-end pipelines: data generation → preprocessing → model training → evaluation.

Integrated GUI and visualization in some projects (e.g., MNIST GUI recognition).

Explored multiple languages (Python & C++) for comparison and understanding of implementation-level differences.

## 🧠 Future Improvements

Refactor old scripts into a unified structure.

Add Jupyter notebooks for step-by-step explanations.

Re-train models with updated architectures.

Add benchmarks for each project.

## 📜 License

Released under the MIT License.
You’re free to use or modify the code with proper credit.