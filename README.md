# Brain Tumor MRI Classification

Automated binary classification of brain MRI scans (`no_tumor` vs. `glioma_tumor`) using traditional machine learning techniques. This project evaluates K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) classifiers, comparing raw pixel intensity representations against **Local Binary Patterns (LBP)** texture descriptors.

---

## 📌 Project Overview

Early detection of brain tumors from MRI slices is critical for effective clinical treatment. While deep learning models achieve high classification accuracy, traditional machine learning approaches offer fast, interpretable, and lightweight alternatives.

---

## 🔬 Methodology

1. **Preprocessing**: Images are converted to grayscale to focus on structural integrity rather than color.
2. **Feature Extraction (LBP)**: A circular LBP operator is applied with a radius of 1 and 8 sampling points using the `uniform` method. This captures local tissue micro-textures, such as edges and spots, which are crucial for distinguishing tumors from healthy tissue.
3. **Histogram Generation**: A normalized 10-bin histogram is generated from the LBP map for each MRI slice.
4. **Feature Caching**: The extracted feature arrays (`X_train`, `y_train`, `X_test`, `y_test`) are saved into a compressed `.npz` file to accelerate subsequent training cycles.
5. **Classification**:
   * **KNN**: Distance-based instance learning using Euclidean distance and majority voting.
   * **SVM**: Maximum-margin hyperplane separation using a Radial Basis Function (RBF) kernel with class balancing to handle potential dataset imbalances.

---

## 🛠️ Project Structure

* `knn.py`: Contains data loading logic, LBP feature extraction, `.npz` caching system, and the custom KNN implementation.
* `svm.py`: Contains the SVM model definition, training loop, evaluation pipeline, and performance reporting.
* `requirements.txt`: List of Python dependencies.

---

## 🚀 Installation & Usage

### Prerequisites
It is recommended to use a virtual environment or Conda:

```bash
conda install -c conda-forge scikit-learn scikit-image opencv numpy tqdm joblib
pip install kagglehub
```
### Running the Project

To train and evaluate the Support Vector Machine (SVM):

```bash
python svm.py
```

To train and evaluate the K-Nearest Neighbors (KNN):

```bash
python knn.py
```

### 📚 References

- Paper : Accurate brain tumor detection using deep convolutional neural network. Computational and Structural Biotechnology Journal. Khan et al. (2022)
- Dataset : Brain Tumor Classification (MRI) by Sartaj Bhuvaji via Kaggle.

