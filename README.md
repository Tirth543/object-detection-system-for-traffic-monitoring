# Object Detection System for Traffic Monitoring

An automated computer vision solution designed for vehicle identification and traffic density analysis through video and image processing.

## Project Overview
This system utilizes deep learning to monitor traffic flow by identifying various vehicle classes. It processes visual data to provide numerical counts and categorizes traffic conditions, assisting in data-driven urban management and infrastructure planning.

## Key Features
- **Multi-Class Identification:** Detects and distinguishes between cars, motorcycles, and trucks.
- **Traffic Density Analysis:** Calculates the number of objects per frame and provides categorical flow status (Normal, Medium, or Heavy).
- **Interactive Web Interface:** A dedicated dashboard for viewing processed outputs, live statistics, and cumulative object counts.

## Technology Stack
- **Deep Learning Architecture:** YOLOv8
- **Core Engine & Inference:** PyTorch (Handles model loading and tensor computations)
- **Backend Framework:** Python & Flask
- **Computer Vision Library:** OpenCV
- **Frontend UI:** HTML5, Tailwind CSS, & JavaScript
