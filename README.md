# CardioScope AI
### AI-Powered ECG Arrhythmia Detection and Analysis

## 1. Project Overview

CardioScope AI is a deep learning-based ECG analysis project designed to identify and classify cardiac arrhythmias from electrocardiogram (ECG) signals.

Electrocardiograms are widely used to monitor the electrical activity of the heart. However, manually analysing large volumes of ECG recordings can be time-consuming and requires specialised expertise.

CardioScope AI explores how artificial intelligence and deep learning can assist in ECG signal analysis by automatically learning patterns from heartbeat signals and identifying different categories of heartbeats.

The project uses a **1D Convolutional Neural Network (1D CNN) with Residual Learning and Temporal Attention Pooling** to analyse ECG signals and classify heartbeat patterns.

It also includes an interactive dashboard that allows users to explore ECG data, understand the model architecture, review evaluation results, and experiment with signal inference.

## 2. Project Objectives

The main objectives of CardioScope AI are:

- To develop a deep learning model for automated ECG heartbeat classification.
- To identify different heartbeat patterns using a 1D CNN architecture.
- To investigate the effectiveness of residual connections in ECG feature extraction.
- To use temporal attention pooling to identify important features across ECG signal segments.
- To evaluate model performance using appropriate classification metrics.
- To provide an interactive dashboard for ECG signal exploration and model analysis.
- To implement a reproducible machine learning pipeline with clearly defined training, validation, and testing datasets.

## 3. Dataset

The project uses the **MIT-BIH Arrhythmia Database**, a publicly available ECG dataset provided by PhysioNet.

Dataset source: https://physionet.org/content/mitdb/1.0.0/

### Dataset Details

- Dataset: MIT-BIH Arrhythmia Database
- Number of records: 48
- Sampling frequency: 360 Hz
- Signal type: Two-channel ambulatory ECG recordings
- Annotation type: Expert-labelled heartbeat annotations
- File formats: WFDB-compatible signal and annotation files

The dataset contains ECG recordings with different heartbeat patterns, including normal beats and several categories of abnormal beats.

### Classification Categories

The project follows an AAMI-style five-class heartbeat classification scheme:

| Class | Description |
|---|---|
| N | Normal and other normal-category beats |
| S | Supraventricular ectopic beats |
| V | Ventricular ectopic beats |
| F | Fusion beats |
| Q | Unknown or paced-category beats |

These categories are used to train the model to distinguish between different heartbeat patterns.

## 4. Data Preprocessing

Raw ECG recordings require preprocessing before they can be used for deep learning.

CardioScope AI follows these preprocessing steps:

**1. ECG Signal Loading**

ECG recordings are loaded using the WFDB Python library.

**2. Signal Filtering**

A bandpass filter between 0.5 Hz and 40 Hz is applied to reduce baseline wander and high-frequency noise.

**3. Lead Selection**

The system prioritises the MLII lead when available. If it is unavailable, it attempts to use lead II, followed by the first available channel.

**4. Heartbeat Segmentation**

Individual heartbeat segments are extracted using the dataset's expert annotations.

Each heartbeat window contains:
- 0.25 seconds before the annotated beat.
- 0.45 seconds after the annotated beat.
- A total of 252 samples at 360 Hz.

**5. Label Assignment**

Each extracted heartbeat is assigned its corresponding classification label.

**6. Data Preparation**

The processed heartbeat segments are converted into arrays suitable for training and evaluation.

## 5. Model Architecture

CardioScope AI uses a custom **1D Convolutional Neural Network with Residual Blocks and Temporal Attention Pooling**.

The architecture is designed to learn meaningful patterns from sequential ECG data.

### Main Components

**1. One-Dimensional Convolutional Layers**

1D convolutional layers extract important local patterns from ECG signals, such as waveform changes and heartbeat characteristics.

**2. Residual Connections**

Residual connections allow information to pass across network layers through shortcut connections.

They help improve information flow during training and support the learning of deeper feature representations.

**3. Batch Normalisation**

Batch normalisation helps stabilise the training process and improve convergence.

**4. Activation Functions**

Non-linear activation functions help the network learn complex patterns from ECG signals.

**5. Temporal Attention Pooling**

Temporal attention pooling assigns different importance weights to different time steps in the extracted feature sequence.

This allows the model to focus on more informative portions of the heartbeat representation.

**6. Classification Layer**

The final classification layer predicts one of the five heartbeat categories:

N, S, V, F, or Q.

### Model Workflow

Raw ECG Signal → Preprocessing → Heartbeat Segmentation → 1D CNN → Residual Blocks → Temporal Attention Pooling → Classification Layer → Predicted Heartbeat Class

## 6. Dataset Splitting Strategy

The project uses a record-based training, validation, and testing strategy rather than randomly splitting individual heartbeat samples.

This is important because heartbeats from the same patient can share similar characteristics. Randomly distributing beats from one recording across training and testing datasets could lead to data leakage and overly optimistic performance results.

The project therefore uses a patient-disjoint evaluation strategy.

- **Training set:** Used to train the neural network.
- **Validation set:** Used to monitor training and support model selection.
- **Testing set:** Used for final evaluation on held-out recordings.

A conservative split is used to avoid including recordings from the same patient in both training and testing. Record 202 is excluded because it is associated with record 201, which belongs to the training group.

This means the evaluation protocol is a modified patient-disjoint version of the commonly used MIT-BIH DS1/DS2 split.

## 7. Model Training

The model is trained using the preprocessed ECG heartbeat segments.

The training pipeline includes:

- Loading the prepared dataset.
- Initialising the 1D CNN model.
- Handling class imbalance using balanced class weights.
- Training the model using labelled heartbeat segments.
- Monitoring performance using the validation dataset.
- Saving the trained model for evaluation and inference.

The model is implemented using PyTorch.

Actual performance values are generated after training and evaluation; no predefined accuracy values are assumed.

## 8. Model Evaluation

The trained model is evaluated using the held-out testing dataset.

The project includes multiple evaluation metrics to provide a broader understanding of classification performance.

### Evaluation Metrics

**Accuracy:** Measures the overall proportion of correctly classified heartbeat samples.

**Precision:** Measures how many samples predicted as a particular class actually belong to that class.

**Recall:** Measures how many actual samples of a particular class are correctly identified.

**F1-Score:** Combines precision and recall into a single metric.

**Macro F1-Score:** Calculates the F1-score independently for each class and gives every class equal importance.

**Confusion Matrix:** Displays the relationship between actual and predicted heartbeat categories and helps identify classification errors.

The project also includes patient-cluster bootstrap confidence interval estimation for macro F1-score to examine uncertainty in the evaluation results.

## 9. Interactive Dashboard

CardioScope AI includes an interactive dashboard developed using Streamlit.

The dashboard provides a user-friendly interface for exploring the project and understanding its different components.

### Dashboard Sections

**1. Overview**

Provides an introduction to CardioScope AI, its purpose, and its main capabilities.

**2. Data Protocol**

Explains the dataset, preprocessing methods, heartbeat segmentation, and patient-disjoint splitting strategy.

**3. Architecture**

Describes the deep learning model, including convolutional layers, residual connections, and temporal attention pooling.

**4. Evaluation**

Displays model evaluation results and performance metrics generated after training.

**5. Signal Explorer**

Allows users to explore ECG signals and examine heartbeat-related data.

**6. Inference**

Provides an interface to experiment with ECG signal inputs and view model predictions.

**7. Research Notes**

Contains additional technical information about the methodology, implementation, and research considerations.

## 10. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning model development |
| NumPy | Numerical computations |
| Pandas | Data handling and analysis |
| SciPy | ECG signal filtering and processing |
| WFDB | Loading MIT-BIH ECG recordings and annotations |
| Scikit-learn | Evaluation metrics and class balancing |
| Streamlit | Interactive dashboard development |
| Matplotlib | Data visualisation |
| Git and GitHub | Version control and project management |

## 11. Project Structure

The project follows a modular structure to separate data processing, model development, evaluation, and dashboard functionality.

```text
cardioscope-ai/
│
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
├── Makefile
│
├── data/
│   └── mitdb/
│
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── prepare.py
│   │   └── splits.py
│   │
│   ├── models/
│   │   └── model.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
│
├── tests/
│
└── .github/
    └── workflows/
```

*Note: The actual repository may contain additional supporting files and modules.*

## 12. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd cardioscope-ai
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

```bash
python -m src.data.download
```

The MIT-BIH Arrhythmia Database is downloaded from PhysioNet.

### Step 5: Prepare the Dataset

```bash
python -m src.data.prepare
```

This processes the ECG recordings and creates the prepared heartbeat dataset.

### Step 6: Train the Model

```bash
python -m src.train
```

This trains the neural network using the prepared training data.

### Step 7: Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard can then be accessed through the local browser address provided by Streamlit.

## 13. Key Features

- Deep learning-based ECG heartbeat classification.
- Custom 1D CNN architecture.
- Residual learning for improved feature extraction.
- Temporal attention pooling.
- ECG signal preprocessing and heartbeat segmentation.
- Five-class heartbeat classification.
- Patient-disjoint dataset evaluation.
- Multiple classification performance metrics.
- Interactive Streamlit dashboard.
- ECG signal exploration and inference interface.
- Modular and reproducible project structure.

## 14. Limitations and Future Improvements

Although CardioScope AI demonstrates the application of deep learning to ECG signal classification, there are several areas where the project can be improved.

### Current Limitations

- The model is trained on the MIT-BIH Arrhythmia Database and may not generalise to every ECG dataset or recording device.
- ECG signal quality and noise can influence classification performance.
- The inference interface is an experimental research feature and does not replace a validated clinical ECG analysis system.
- The model requires appropriate heartbeat segmentation, consistent with its training data.

### Future Improvements

- Expand training using additional publicly available ECG datasets.
- Investigate more advanced architectures such as CNN-LSTM and Transformer-based models.
- Improve generalisation across different patients and ECG acquisition devices.
- Introduce real-time ECG signal processing.
- Integrate explainability techniques to better understand model predictions.
- Explore mobile and cloud-based deployment.
- Improve the inference pipeline to support automated heartbeat localisation.

## 15. Applications

Potential research and educational applications include:

- Automated ECG signal analysis.
- Heartbeat pattern recognition.
- Arrhythmia classification research.
- Biomedical signal processing.
- Deep learning experimentation.
- Educational demonstrations of AI in healthcare.

## 16. Disclaimer

CardioScope AI is an academic and research-oriented prototype developed for educational and experimental purposes.

It is not a certified medical device and has not been validated for clinical use. Its predictions must not be used for medical diagnosis, treatment decisions, emergency triage, or patient monitoring.

## 17. References

1. Moody, G. B., and Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*.
2. MIT-BIH Arrhythmia Database, PhysioNet: https://physionet.org/content/mitdb/1.0.0/
3. WFDB Python Package Documentation: https://wfdb.readthedocs.io/
4. PyTorch Documentation: https://pytorch.org/docs/
5. Scikit-learn Documentation: https://scikit-learn.org/stable/

---

**CardioScope AI — Exploring the intersection of Artificial Intelligence, Deep Learning, and Biomedical Signal Processing.**