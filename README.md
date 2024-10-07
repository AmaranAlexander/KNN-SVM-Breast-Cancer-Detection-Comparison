
# Breast Cancer Detection: KNN vs. SVM

This repository contains the code and research paper comparing the accuracy of two machine learning algorithms, **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)**, in detecting breast cancer. The project is based on datasets from the **California Irvine Machine Learning Repository**.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Algorithms](#algorithms)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Support Vector Machines (SVM)](#support-vector-machines-svm)
- [Results](#results)
- [Paper](#paper)
- [How to Run the Code](#how-to-run-the-code)
- [References](#references)

## Introduction

Breast cancer is one of the leading causes of death among women globally. Accurate and early detection is critical in increasing survival rates. This project compares the accuracy of two supervised machine learning algorithms—**KNN** and **SVM**—to see which algorithm is more effective in detecting malignant breast cancer cells.

The paper accompanying this repository explores the implementation, performance, and evaluation of these two algorithms on the breast cancer dataset.

## Dataset

The dataset used in this project is from the **California Irvine Machine Learning Repository** and contains 30 different features derived from breast mass images. The features are based on properties such as texture, perimeter, area, compactness, and more.

- **Target**: Benign (0) or Malignant (1)
- **Number of Features**: 30

## Algorithms

### K-Nearest Neighbors (KNN)

The KNN algorithm classifies data points based on the nearest neighbors. The K-value (number of neighbors to vote from) plays a critical role in the accuracy of the KNN algorithm.

### Support Vector Machines (SVM)

SVM is a classification algorithm that works by finding a hyperplane that best separates the data points into classes. The kernel function in SVM helps increase the accuracy by handling non-linearly separable data.

## Results

- **KNN Accuracy**: 95.6%
- **SVM Accuracy**: 97.4%
  
From the results, we conclude that **SVM** is more accurate than **KNN** for this particular dataset, with a 1.8% margin in accuracy.

## Paper

The full paper explaining the methodology, implementation, and results can be found [here](./KNN-SVM-Breast-Cancer-Detection.pdf). It discusses the background of both algorithms, how they were implemented, and a detailed evaluation of their performance in detecting breast cancer.

## How to Run the Code

### Requirements

- Python 3.x
- Scikit-learn
- NumPy
- Google Colab (optional, to run the code in the browser)

### Running the Program

1. Clone the repository:
    ```bash
    git clone https://github.com/AmaranAlexander/KNN-SVM-Breast-Cancer-Detection-Comparison.git
    cd KNN-SVM-Breast-Cancer-Detection-Comparison
    ```

2. Install the dependencies:
    ```bash
    pip install scikit-learn numpy
    ```

3. Run the code in a Python environment or Google Colab:
    - For **KNN**: `knndetection.py`
    - For **SVM**: `svmdetection.py`

4. The program will output the accuracy of each algorithm based on the dataset.

### Sample Commands:

```bash
python knndetection.py
python svmdetection.py
```

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Breast Cancer Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [Machine Learning: KNN vs SVM](./KNN-SVM-Breast-Cancer-Detection.pdf)
