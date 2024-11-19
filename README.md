# Vehicle Detection and Classification System

This project aims to develop a **Vehicle Detection and Classification System** that utilizes state-of-the-art machine learning techniques. By employing the **YOLO (You Only Look Once)** object detection algorithm, the system is capable of detecting various types of vehicles in real-time. The detected vehicles are then classified into different categories (such as car, bus, truck, etc.) using a Convolutional Neural Network (CNN). The system also utilizes **transfer learning** for improved model performance and **Generative Adversarial Networks (GANs)** to generate synthetic vehicle images, enriching the dataset and improving the accuracy of the detection model.

## Abstract

The Vehicle Detection and Classification System leverages **YOLO** for real-time vehicle detection, a **CNN model** for vehicle classification, and **transfer learning** for performance enhancement. Additionally, **GANs** are used to generate synthetic vehicle images to augment the training dataset. The system is designed to automatically detect and classify vehicles from images, making it suitable for traffic monitoring, autonomous driving, and vehicle tracking applications.

The dataset used for training contains various vehicle images with labeled annotations, and the system is capable of detecting and classifying vehicles into different categories. The integration of GANs helps mitigate class imbalances by augmenting the dataset with synthetic images of vehicles.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Design](#project-design)
3. [Module Description](#module-description)
4. [Results](#results)
5. [Conclusion](#conclusion)

## Screenshots

Here are some screenshots from the project:

- ![Result 4](https://github.com/aditiaherr/Vehicle-Detection-and-Classification-System/blob/main/result_4.jpeg)
- ![Result 6](https://github.com/aditiaherr/Vehicle-Detection-and-Classification-System/blob/main/result_6.jpeg)

## Chapter 1: Introduction

### Problem Statement

In modern transportation systems, vehicle detection and classification play a crucial role in traffic monitoring, autonomous driving, and vehicle tracking. Existing systems often struggle with real-time detection and classification, especially in complex and dynamic environments. This project addresses these challenges by utilizing YOLO for efficient vehicle detection, CNN for accurate classification, and GANs to augment the dataset for improved model performance.

### Project Idea

This project integrates cutting-edge machine learning techniques to detect and classify vehicles in real-time from images. The system is designed to be used in applications such as traffic management systems, autonomous vehicles, and security surveillance.

### Motivation

The motivation behind this project is to improve vehicle detection and classification accuracy for various applications, including traffic monitoring, autonomous driving, and security surveillance. By utilizing advanced techniques such as YOLO, transfer learning, and GANs, the system can achieve higher accuracy and efficiency.

### Scope

- Detect and classify vehicles using YOLO and CNN.
- Utilize transfer learning to enhance the model's performance.
- Generate synthetic vehicle images using GANs to augment the dataset.
- Provide real-time vehicle detection and classification in dynamic environments.

### Literature Survey / Requirement Analysis

A review of existing vehicle detection systems reveals that YOLO-based solutions provide fast and efficient detection in real-time. However, many systems fail to achieve high classification accuracy, especially in challenging scenarios such as overlapping vehicles or poor lighting conditions. This project aims to address these limitations by using a combination of YOLO, CNN, transfer learning, and GANs.

## Chapter 2: Project Design

### Hardware & Software Requirements

#### Hardware:
- Computer with at least 8 GB RAM and a dedicated GPU for model training and inference.
- Camera or webcam for capturing real-time images.

#### Software:
- Python for development.
- YOLO for vehicle detection.
- CNN for vehicle classification.
- GAN for generating synthetic images.
- OpenCV for image processing.
- TensorFlow/Keras for model training.

### Dataset Design

The dataset consists of images of various vehicles, including cars, trucks, buses, and motorcycles. The images are annotated with bounding boxes and labels to train the YOLO object detection model. The dataset is augmented using GANs to generate synthetic images, addressing class imbalances and improving model generalization.

### Hours Estimation

- Dataset Creation and Annotation: 30 hours
- Model Training and Tuning: 40 hours
- Transfer Learning Implementation: 20 hours
- GAN Training for Image Augmentation: 30 hours
- Testing and Documentation: 15 hours

**Total Estimated Hours**: 135 hours

## Chapter 3: Module Description

### Block Diagram

1. **User Interface**: Captures images or video frames from a camera for vehicle detection.
2. **Image Capture**: Captures real-time images of vehicles.
3. **YOLO Object Detection Model**: Detects vehicles and provides bounding boxes around detected vehicles.
4. **CNN Classification Model**: Classifies the detected vehicles into categories (e.g., car, bus, truck).
5. **Synthetic Image Generation (GAN)**: Augments the dataset with synthetic vehicle images to improve model performance.
6. **Vehicle Detection and Classification Output**: Displays the detected vehicles along with their respective classifications.

### System Architecture

The YOLOv5 architecture is employed for real-time vehicle detection. YOLOv5's efficiency and accuracy make it suitable for real-time applications. The system also utilizes a CNN for vehicle classification, which is trained using transfer learning from a pre-trained model to improve the model's generalization capabilities.

## Chapter 4: Results

### Screenshots

- ![Result 4](https://github.com/aditiaherr/Vehicle-Detection-and-Classification-System/blob/main/result_4.jpeg)
- ![Result 6](https://github.com/aditiaherr/Vehicle-Detection-and-Classification-System/blob/main/result_6.jpeg)


## Chapter 5: Conclusion

This project successfully implements a vehicle detection and classification system using YOLO for real-time object detection, CNN for classification, and GANs for data augmentation. The system is designed to be used in applications like traffic monitoring, autonomous driving, and security surveillance. Future work could include improving classification accuracy in challenging environments and integrating the system into real-world applications.
