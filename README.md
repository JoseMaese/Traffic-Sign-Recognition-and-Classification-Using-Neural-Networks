<br />
<div align="center">
  <a href="https://github.com/JoseMaese/Traffic-Sign-Recognition-and-Classification-Using-Neural-Networks">
    <img src="images/logo.png" alt="Logo" width="170" height="100">
  </a>
<h3 align="center">Traffic-Sign-Recognition-and-Classification-Using-Neural-Networks</h3>
  <p align="center">
    José Enrique Maese Álvarez
    <br />
  </p>
</div>


## Overview

This project focuses on the application of neural networks for the identification and classification of traffic signs using computer vision techniques. The primary models used are YOLOv8 for detection and ResNet-50 for classification, aimed at improving road safety and advancing autonomous vehicle technologies.

## Table of Contents
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li> <a href="#Overview">Overview</a> </li>
    <li> <a href="#Objectives">Objectives</a> </li>
    <li><a href="#Methodology">Methodology</a></li>
    <li><a href="#Results">Results</a></li>
    <li><a href="#Conclusions">Conclusions</a></li>
    <li><a href="#Future Improvements">Future Improvements</a></li>
  </ol>
</details>


## Objectives

### General Objectives
- Develop and evaluate a system for the recognition and classification of traffic signs in real-world environments using neural networks.
- Gain proficiency in neural network environments and cloud-based training.
- Manage large datasets and implement models using various programming languages.

### Specific Objectives
- Create a model capable of locating and segmenting traffic signs.
- Develop a model to classify cropped images of traffic signs into detailed categories.
- Manage the necessary databases for training the neural models.
- Integrate both processes into a comprehensive program capable of analyzing videos.
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Methodology

### Tools
- **Programming Language:** Python
- **Development Environments:** Google Colab, Spider, Visual Studio
- **Database Management:** Roboflow, Kaggle, GitHub
- **Result Analysis:** Weights & Biases (wandb)

### Procedure
1. **Data Acquisition and Preprocessing:** Collect and preprocess images from the front of a vehicle.
2. **Model Training:**
   - Train a YOLO model to detect and classify traffic signs into broad categories.
   - Train a ResNet-50 model to classify segmented images into specific classes.
3. **Model Integration:** Develop a program to run both models consecutively on video data.
4. **Real-world Testing:** Evaluate the system using videos captured in real-world driving conditions.

<div align="center">
  <img src="imagenes/esquema proyecto.png" alt="esquema proyecto.png" width="960" height="540">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results

### YOLOv8 Detection
- **Architecture:** Optimized for fast and accurate detection of traffic signs.
- **Performance Metrics:** Achieved high precision and recall rates across various classes (danger, mandatory, prohibitory and other).
- 
<div align="center">
  <img src="imagenes/map50.png" alt="mAP50" width="1264" height="664">
</div>

<div align="center">
  <img src="imagenes/modelo004_matriz.png" alt="Matriz de confusion" width="1500" height="1125">
</div>

### ResNet-50 Classification
- **Function:** Refines detection results by classifying traffic signs into specific subclasses.
- **Performance:** Demonstrated high accuracy but identified areas for improvement in secondary classification.

<div align="center">
  <img src="imagenes/train_validation_resnet_75epocas.png" alt="tran and validation resNet-50" width="720" height="496">
</div>

### Real-world Testing
- The system was tested with videos under different lighting conditions and resolutions, showing robust performance.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Conclusions

- The combined use of YOLOv8 and ResNet-50 models effectively identifies and classifies traffic signs in real-world conditions.
- Cloud computing platforms significantly optimized model training times.
- The system's robustness in varying conditions suggests its applicability in real-world driving scenarios.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Future Improvements

- **Database Enhancement:** Improve training and validation datasets for better model performance.
- **Output Processing:** Further refine the processing of model outputs to enhance accuracy.
- **Secondary Classification:** Explore more advanced models for secondary classification tasks.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

