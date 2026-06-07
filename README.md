# Handwritten Digit Recognition Using Deep Learning

## Overview

This project implements a Handwritten Digit Recognition System using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset and can accurately recognize handwritten digits from 0 to 9.

The project demonstrates the application of Deep Learning and Computer Vision in image classification and pattern recognition.

---

## Objective

The primary objective of this project is to build an intelligent system capable of recognizing handwritten digits with high accuracy using Deep Learning techniques.

---

## Dataset

The project uses the **MNIST Handwritten Digits Dataset**.

### Dataset Information

* Training Images: 60,000
* Testing Images: 10,000
* Image Size: 28 × 28 Pixels
* Classes: 10 (Digits 0–9)

The dataset consists of grayscale images of handwritten digits collected from different individuals.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn

---

## Deep Learning Architecture

The model uses a Convolutional Neural Network (CNN) consisting of:

### Feature Extraction Layers

* Convolution Layer (32 Filters)
* Max Pooling Layer
* Convolution Layer (64 Filters)
* Max Pooling Layer

### Classification Layers

* Flatten Layer
* Dense Layer (128 Neurons)
* Dropout Layer
* Output Layer (10 Classes)

---

## Project Workflow

1. Load MNIST Dataset
2. Normalize Image Data
3. Build CNN Architecture
4. Train the Model
5. Evaluate Performance
6. Generate Accuracy and Loss Graphs
7. Predict Handwritten Digits
8. Save Trained Model

---

## Model Performance

The model achieves high accuracy on the test dataset and successfully recognizes handwritten digits.

Evaluation Metrics:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

---

## Generated Outputs

The following files are generated:

* training_performance.png
* sample_prediction.png
* handwritten_model.h5

---

## Sample Prediction

Input: Handwritten Digit Image

Predicted Output: Digit 7

Model Confidence: 98%+

---

## Applications

* Bank Cheque Processing
* Postal Code Recognition
* Automated Form Processing
* Educational Tools
* Digital Document Analysis

---

## Future Enhancements

* Real-Time Webcam Digit Recognition
* Mobile Application Deployment
* Character and Alphabet Recognition
* Multi-Language Handwriting Recognition
* Cloud-Based Prediction System

---

## Conclusion

This project successfully demonstrates the power of Deep Learning in image classification tasks. Using a Convolutional Neural Network, the system can accurately recognize handwritten digits and serves as a foundation for more advanced computer vision applications.

---

## Author

**Prachith Balaji G**

Machine Learning Intern – CodeAlpha
