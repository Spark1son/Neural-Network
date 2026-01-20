# Neural Network Project – CIFAR-10 Classification

This project explores image classification on the CIFAR-10 dataset using a combination of **deep learning** and **classical machine learning** techniques.

Specifically, a **Convolutional Neural Network (CNN)** is used as a feature extractor, and a **K-Nearest Neighbors (KNN)** classifier is applied on the extracted features.  
The goal is to demonstrate practical understanding of model design, feature representation, and evaluation.

---

## Project Objectives

- Understand how CNNs extract high-level image features
- Apply a classical ML algorithm (KNN) on learned representations
- Compare deep feature-based classification with traditional methods
- Practice clean, reproducible ML experimentation

---

## Technologies & Skills Demonstrated

- **Python**
- **PyTorch** (model definition, dataset handling)
- **Torchvision** (CIFAR-10, transforms)
- **Scikit-learn** (KNN, evaluation metrics)
- **Jupyter Notebook**
- Machine Learning & Deep Learning fundamentals
- Feature extraction and model evaluation

---

## Notebook Overview

- **KNN_CNN.ipynb**: Loads CIFAR-10, defines a CNN, extracts features, and applies KNN for classification. Includes performance evaluation and analysis.
- **rbf.ipynb**: Implements Radial Basis Function (RBF) networks for classification, including feature extraction and clustering techniques.
- **mlp_hinge_loss_implementation.ipynb**: Multi-Layer Perceptron (MLP) implementation using hinge loss, with training and evaluation on CIFAR-10.
- **svm_reference_implementation.ipynb**: SVM classification using scikit-learn, with feature extraction and model evaluation.
- **svm_scratch_implementation.ipynb**: Custom SVM implementation from scratch, demonstrating algorithmic fundamentals and manual optimization.

---

## Reports

The `Reports/` directory contains PDF reports for each method:
- **Neural_RBF.pdf**: RBF network experiments and results.
- **Neural-MLP.pdf**: MLP experiments and results.
- **Neural-neighbors.pdf**: KNN and CNN feature extraction experiments.
- **Neural-part_2.pdf**: Additional or combined results and analysis.

---

## Dataset

- **CIFAR-10**
  - 60,000 RGB images (32×32)
  - 10 object categories
  - Automatically downloaded via Torchvision

No dataset files are stored in this repository.

---

## How to Run

1. Clone the repository:
	```bash
	git clone https://github.com/Spark1son/Neural-Network.git
	cd Neural-Network
	```
2. Install dependencies (recommended: use a virtual environment):
	```bash
	pip install torch torchvision scikit-learn numpy matplotlib
	```
3. Open any notebook (e.g., `KNN_CNN.ipynb`) in Jupyter or VS Code and run the cells.
4. The datasets will be downloaded automatically on first run.

---

## File Naming Conventions

All notebooks use clear, descriptive, and professional names for easy identification:
- `KNN_CNN.ipynb`
- `rbf.ipynb`
- `mlp_hinge_loss_implementation.ipynb`
- `svm_reference_implementation.ipynb`
- `svm_scratch_implementation.ipynb`

---

## Contact

For questions or contributions, please open an issue or contact the repository maintainer.
