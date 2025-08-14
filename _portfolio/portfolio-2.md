---
title: "Object Detection using YOLO Model integrated with ROS2 for Autonomous Robot"
excerpt: "<br/><img src='/images/abu_object_detection.png'>"
collection: portfolio
---

![Object Detection ROS2](/images/abu_object_detection.png)

# Table of Contents
- [I. Introduction](#i-introduction)
  - [1.1. Game Rule](#11-game-rule)
  - [1.2. Responsibilities](#12-responsibilities)
- [II. Methodology](#ii-methodology)
  - [2.1. Dataset Preparation](#21-dataset-preparation)
  - [2.2. Model Evaluation](#22-model-evaluation)
  - [2.3. Hardware Implementation](#23-hardware-implementation)
- [III. Results](#iii-results)
- [IV. Conclusion](#iv-conclusion)


---

# I. Introduction
The primary objective of this project is to participate in the ABU Robocon Competition 2024 by performing object detection (system) to identify blue and red balls. It will allow a robot to complete challenges autonomously under the rules of the competition.

![ABU 2024](/images/abu_01.png)

## 1.1. Game Rule
In the competition each teams need to have two robot with different functions and different responsibility. For 1st Robot, the robot that only works in Area 1 and Area 2 Robot 1 is either Manual Robot or Automatic Robot.  Manual  Robot:  The  robot  which  is  operated  by  operator  via wireless connection. For 2nd Robot, must be an Automatic Robot fully work on Area 3 by starting move from Area 1 to Area 2 after arrived in Area 3 it will detect the color of ball and catch the ball to the silos.

## 1.2. Responsibilities
In my works, I'm a lead of programming team for Robot 2 (Autonomous Robot). The function of the Robot 2 including:
- Used STM32 for firmware (Control Motor, ...) then it will publish the position and velocity of robot thorugh CANBUS communication protocol to Mini PC (Jetson AGX Xavier) via ROS2 (Foxy) frameworks. After the robot 2 move autonomous to Area 3 it will perform object detection to detect the color of the ball (if we are in blue team the robot 2 will catch only the blue ball) and bring them to silos.
- Used Realsense D435i for inference and the detection model are execute on Jetson AGX Xavier.
- The Robot 2 can detect the color of ball and counting the ball on silos and know which ball is in the shorter distance.

---

# II. Methodology

## 2.1. Dataset Preparation
We used YOLOv8 Pre-trained modeland find-tuned to perform our object detection task.

![Data Preparation](/images/Dataset_Prep.png)

- Dataset collect: focuses on scenarios found in Robocon 2024 competition
game-fields, featuring blue and red balls. It covers various challenging conditions encountered
during competitions situation, including the area for blue ball and area for red ball.
- Annotations: After preparing our dataset for each condition needed, the next stage involves annotating the images. We chose Roboflow for this task because it’s a widely recognized and user-friendly
framework, especially effective for labeling images in YOLO dataset training.

## 2.2. Model Evaluation
We used Ultralytics packages YOLOv8.2.16, python3.10.12 torch-2.2.2+cu121 CUDA:0 (Nvidia Geforce RTX 3050, VRAM = 3902MB). As shown in image below, the model performance across all metrics with high precision, recall, and mAP values. This indicates that the model is generally effective at detecting both blue and
red balls.
- Precision: Indicating the model’s ability to identify objects accurately. The precision for blue balls (0.93) is slightly higher than for red balls (0.923).
- Recall: Representing the model’s ability to detect all objects. The recall for both of ball (0.925) are indicated the model is better at identifying.
- Mean Average Precision (mAP): Perform measure that combines accuracy, precision, and
recall to evaluate the model’s object detection capabilities across multiple frames.
- mAP50-95: Check accuracy at IoU levels from 0.5 to 0.95 (High mean better). Inter-
estingly, the mAP 50-90 for red balls (0.691) is higher than for blue balls (0.679).

![precision](/images/precision.png)

## 2.3. Hardware Implementation
In this project, we implemented on the Jetson AGX Xavier using the RealSense D435i. We flashed Ubuntu 20.04 on Jetson through Jetpack 5.1.2 buit-in Libraries:
- CUDA: 11.4.315
- TensorRT: 8.5.2.2
- CuDNN: 8.6.0.166
- OpenCV: 4.5.4 

Setup Realsense for Jetson Packages:
- Ultralytics: 8.2.16 
- Install pytorch and torchvision (Available with JP 5.1.2)
- Setup onnx for tensor RT conversion 

# III. Results
As shown in figure below, including:
- **Inference Time:** The model processes each input in **21.8 ms** (see *Figure below*). This allows for quick and accurate detection, which is essential in a fast-paced environment like **Robocon**.
- During object detection (see *Figure below*), the **GPU** runs at **79% usage** and maintains a stable temperature of **40°C**. This ensures the system can handle **high-resolution images** and **multiple objects** without slowing down.
- The **CPU** usage is **38.6%** during detection (see *Figure below*), leaving enough processing power for other tasks. This makes it suitable for running **ROS2 control nodes** to manage the robot’s motion in real time, ensuring smooth and responsive behavior in Robocon matches.

![deployment](/images/performance.png)

# IV. Conclusion
This project aims to enable the robot to automatically catch and deliver balls to silos within 3 minutes for a Robocon game.

Key points:
- **GPU usage (79–90%)** shows efficient use for real-time object detection with YOLOv8.
- **CPU usage (38%)** leaves room for other tasks, such as motion control using ROS2.
- **Fast inference time (21.8 ms)** ensures quick and accurate detection, crucial for dynamic gameplay.
- The system meets real-time requirements, allowing the robot (MR2) to respond promptly.

✅ Overall, the model runs efficiently without causing slowdowns, making it suitable for competition use.