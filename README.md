# EEG-based Passive Brain-Computer Interface (BCI) for Attention State Detection

This repository contains code and documentation for a project focused on classifying human mental attention states—**focused**, **unfocused**, and **drowsy**—using EEG data and machine learning. This project was developed as part of the **CSCI 490** course at **Nazarbayev University**, School of Engineering and Digital Sciences. The goal was to design a **passive BCI system** that can monitor human attention in real-time, with applications in safety-critical fields such as transportation and surveillance.

## Project Overview

I implemented a **Passive Brain-Computer Interface (BCI)** system that classifies three mental states based on EEG data:

- **Focused**: High attention and alertness.
- **Unfocused**: Reduced attention, passive engagement.
- **Drowsy**: Low alertness, near-sleep or light dozing.

This project aims to detect subtle mental states that can lead to safety risks if undetected. By using machine learning on EEG signals, the system can provide real-time alerts or take adaptive actions when an operator’s attention falters, especially in monotonous tasks such as driving or operating autopilot systems.

## Dataset

- **Source**: Kaggle - EEG Data for Mental Attention State Detection
- https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection
- **Data Description**: The dataset contains **25 hours of EEG recordings** from **5 participants**, each contributing approximately **5 hours** of data. The recordings were collected during a monotonous simulated train-driving task, designed to evoke different levels of attention.
- **Channels**: 12 EEG channels recorded from various brain regions (e.g., frontal, parietal lobes).
- **Class Imbalance**: The dataset exhibits a significant class imbalance with more **unfocused** states than **focused** and **drowsy** states. This posed a challenge in ensuring accurate detection of less frequent classes.

## Key Challenges and Solutions

### 1. **Class Imbalance**
   - **Problem**: The dataset had far more **unfocused** samples than the other states, leading to biased model performance.
   - **Solution**: **Class balancing techniques** including oversampling of minority classes and **class weight adjustments** were applied during training. This significantly improved the detection of **drowsy** states, which initially had poor recall.

### 2. **Feature Selection and Dimensionality Reduction**
   - **Problem**: EEG data is high-dimensional, making it computationally expensive to process.
   - **Solution**: I used **Principal Component Analysis (PCA)** to reduce the dimensionality of the data while retaining **95% of the variance**. This allowed for faster training and reduced the risk of overfitting while maintaining the key characteristics of EEG signals.

### 3. **Noisy Data**
   - **Problem**: Raw EEG data often includes noise and artifacts, which can lead to inaccurate predictions.
   - **Solution**: **Band-pass filtering** was applied to remove noise outside the typical EEG frequency range (1–50 Hz), ensuring cleaner signals for classification.

### 4. **Subtle Differences Between States**
   - **Problem**: The boundary between **unfocused** and **drowsy** states is subtle, and these states often overlap in EEG patterns, making accurate classification challenging.
   - **Solution**: Several **machine learning models** were experimented (including **RLDA**, **SVM**, **Random Forest**, and **Logistic Regression**) and found that **Regularized Linear Discriminant Analysis (RLDA)** offered the best performance for distinguishing these nuanced mental states.

## Methods

### Preprocessing Steps:
- **Noise Filtering**: Applied a **band-pass filter** to retain only the relevant EEG frequency ranges.
- **Feature Scaling**: Used **StandardScaler** to normalize the EEG data, ensuring comparability between different channels.
- **Dimensionality Reduction**: Applied **PCA** to reduce feature space, improving computational efficiency.
- **Class Balancing**: Employed oversampling and weight adjustments to balance the dataset, improving model performance on the minority classes.

### Machine Learning Models:
- **Support Vector Machine (SVM)** with RBF Kernel
- **Regularized Linear Discriminant Analysis (RLDA)**
- **Random Forest**
- **Logistic Regression** (Ridge and Lasso)

### Performance Metrics:
Each model was evaluated using **F1-scores** for each class (**focused**, **unfocused**, **drowsy**) and the **average F1-score** across all classes.

## Results

| Model                    | Focused F1 | Unfocused F1 | Drowsy F1 | Avg F1 |
|--------------------------|------------|--------------|-----------|--------|
| RLDA                     | 0.80       | 0.79         | 0.69      | 0.76   |
| Random Forest            | 0.79       | 0.75         | 0.63      | 0.72   |
| SVM (RBF Kernel)         | 0.75       | 0.72         | 0.51      | 0.66   |
| Logistic Regression (Ridge) | 0.71    | 0.64         | 0.50      | 0.62   |
| Logistic Regression (Lasso) | 0.70    | 0.67         | 0.52      | 0.63   |

The **RLDA** model outperformed others, achieving the highest F1-scores for all classes, with an average F1 of **0.76**.

## Discussion

- **Focused vs. Unfocused States**: These states were relatively easy to distinguish, with models like **RLDA** and **Random Forest** achieving strong performance in detecting them.
- **Unfocused vs. Drowsy**: The distinction between these two states proved challenging due to their subtle EEG differences. Despite this, **RLDA** achieved the best results, demonstrating the importance of regularization in handling complex EEG patterns.

## Limitations and Future Work

- **Dataset Limitations**: The dataset had limited **participant diversity** and missing sessions, which may affect generalizability. Additionally, **missing EEG channels** might have impacted model performance.
- **Modeling Constraints**: We did not explore deep learning approaches due to the limited size of the dataset. Future work could focus on **temporal modeling** techniques (e.g., **LSTMs**) to better capture sequential dependencies in EEG data.

The project was inspired by:

Inan, C., Kaya, M., & Mishchenko, Y. (2019). Distinguishing mental attention states of humans via an EEG-based passive BCI using machine learning methods. *Expert Systems with Applications, 134*, 153–166.

---

This README will present your project effectively and provide a comprehensive understanding of your work, including its challenges, methods, and results. You can simply copy-paste this into your GitHub repository’s `README.md` file!
