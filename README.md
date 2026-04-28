# 🚦 Object Detection System for Traffic Monitoring

An automated computer vision solution developed to identify vehicles and analyze traffic flow density through advanced deep learning techniques.

## 📌 Project Overview
This project addresses urban traffic challenges by providing a digital tool for vehicle identification and density categorization. By processing video and image inputs, the system identifies specific vehicle classes and provides statistical insights to assist in infrastructure planning and traffic management.

## 🚀 Key Features
* **Intelligent Identification:** Detects and classifies Cars, Motorcycles, and Trucks with high precision.
* **Density Analysis:** Automatically calculates object load per frame and assigns a traffic status: **Normal Flow 🟢**, **Medium Traffic 🟡**, or **Heavy Traffic 🔴**.
* **Web-Based Dashboard:** A minimalist, glassmorphism-themed UI for seamless media upload and visualization of detection results.
* **Inference Optimization:** Supports GPU acceleration via PyTorch for faster processing on compatible hardware.

## 🛠️ Technology Stack
* **AI Engine:** YOLOv8 (You Only Look Once) Architecture
* **Deep Learning Backend:** PyTorch (Handles tensor computations and model inference)
* **Backend Framework:** Python Flask
* **Image Processing:** OpenCV
* **Frontend UI:** HTML5, Tailwind CSS, & JavaScript

## 📂 Project Structure
```text
├── app.py              # Flask server and AI inference logic
├── best.pt             # Trained YOLOv8 model weights
├── templates/          # Frontend assets
│   └── index.html      # Main dashboard UI
├── static/             # Static resource management
│   ├── uploads/        # Input media storage
│   └── results/        # Processed output storage
└── README.md           # Documentation
