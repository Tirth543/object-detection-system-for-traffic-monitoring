# 🚦 Object Detection System for Traffic Monitoring

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)

An automated computer vision solution developed to identify vehicles and analyze traffic flow density through advanced deep learning techniques.

## 📌 Project Overview
This project addresses urban traffic challenges by providing a digital tool for vehicle identification and density categorization. By processing video and image inputs, the system identifies specific vehicle classes and provides statistical insights to assist in infrastructure planning and traffic management.

## 🚀 Key Features
* **Intelligent Identification:** Detects and classifies Cars, Motorcycles, and Trucks with high precision.
* **Density Analysis:** Automatically calculates object load per frame and assigns a traffic status: **Normal Flow 🟢**, **Medium Traffic 🟡**, or **Heavy Traffic 🔴**.
* **Web-Based Dashboard:** A minimalist, glassmorphism-themed UI for seamless media upload and visualization of detection results.
* **Inference Optimization:** Supports GPU acceleration via PyTorch for faster processing on compatible hardware.

## 🛠️ Technology Stack
* **AI Engine:** YOLOv8 Architecture
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
```

## ⚙️ Installation & Setup

Follow these steps to run the project locally on your machine:

### 1. Prerequisites
Ensure you have **Python 3.8 or higher** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/Tirth543/object-detection-system-for-traffic-monitoring.git
cd object-detection-system-for-traffic-monitoring
```
### 3. Install Dependencies
Install the required Python libraries using pip:
```bash
pip install flask ultralytics opencv-python torch torchvision
```
### 4. Launch the Application
Start the Flask server by running:
```bash
python app.py
```
### 5. Access the Dashboard
Once the server is running, open your web browser and go to:
`http://127.0.0.1:3000`

---

## 🛠️ Project Workflow

The system follows a structured pipeline to process traffic data:

1. **Input Stage:** The user uploads an image or video through the web dashboard.
2. **Inference Stage:** The **YOLOv8** model processes the frames using the **PyTorch** backend to detect and classify vehicles.
3. **Logic Stage:** The system calculates the total vehicle count and determines the traffic density status (Normal, Medium, or Heavy).
4. **Output Stage:** The processed media is saved in the `static/results/` folder and displayed on the UI with bounding boxes and real-time statistics.

---
