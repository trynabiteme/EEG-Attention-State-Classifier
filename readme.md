# EEG-based Passive Brain-Computer Interface (BCI) for Attention State Detection

This repository contains code and documentation for a project focused on classifying human mental attention states—focused, unfocused, and drowsy—using EEG data and machine learning. The goal was to design a passive BCI system that can monitor human attention in real-time, with applications in safety-critical fields such as transportation and surveillance.

## Project Overview

I implemented a Passive Brain-Computer Interface (BCI) system that classifies three mental states based on EEG data:

- **Focused**: High attention and alertness
- **Unfocused**: Reduced attention, passive engagement
- **Drowsy**: Low alertness, near-sleep or light dozing

This project aims to detect subtle mental states that can lead to safety risks if undetected. By using machine learning on EEG signals, the system can provide real-time alerts or take adaptive actions when an operator's attention falters, especially in monotonous tasks such as driving or operating autopilot systems.

## Dataset

- **Source**: [Kaggle - EEG Data for Mental Attention State Detection](https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection)
- **Data Description**: The dataset contains EEG recordings from 23 subjects, each performing a monotonous simulated train-driving task designed to evoke different levels of attention
- **Channels**: 14 EEG channels recorded from various brain regions (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
- **Sampling Rate**: 128 Hz
- **Class Distribution**: 
  - Focused: 3,450 samples (20%)
  - Unfocused: 10,250 samples (60%)
  - Drowsy: 3,450 samples (20%)

## Key Challenges and Solutions

### 1. Class Imbalance
- **Problem**: The dataset exhibited a 3:1 class imbalance with significantly more unfocused samples than focused and drowsy states
- **Solution**: Applied strategic undersampling to balance the dataset from 17,150 to 10,350 samples, ensuring equal representation across all three classes

### 2. Signal Processing and Noise Reduction
- **Problem**: Raw EEG data contains noise and artifacts across different frequency bands
- **Solution**: Implemented band-pass filtering to isolate four key brainwave frequencies:
  - Delta (0.5-4 Hz)
  - Theta (4-8 Hz)
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)

### 3. High Dimensionality
- **Problem**: Each 4-second sample contained 7,168 features (512 time points × 14 channels)
- **Solution**: Applied PCA retaining 95% variance, reducing features to ~1,200 dimensions—achieving 6x dimensionality reduction while preserving critical EEG patterns

### 4. Subtle State Distinctions
- **Problem**: The boundary between unfocused and drowsy states is subtle with overlapping EEG patterns
- **Solution**: Evaluated 7 different machine learning models, finding that ensemble methods (XGBoost, CatBoost) provided the best discrimination between these nuanced states

## Methods

### Preprocessing Pipeline:
1. **StandardScaler Normalization**: Normalized EEG data across channels to ensure equal contribution
2. **Band-pass Filtering**: Extracted frequency-specific brain rhythms known to correlate with attention states
3. **Windowing**: Segmented continuous EEG into 4-second windows with consensus labeling
4. **Vectorization**: Flattened 3D data for sklearn compatibility
5. **PCA Dimensionality Reduction**: Reduced features from 7,168 to ~1,200 (95% variance retained)

### Machine Learning Models:
- Regularized Linear Discriminant Analysis (RLDA) with shrinkage=0.90
- Random Forest (n_estimators=100)
- Support Vector Machine (RBF kernel)
- Logistic Regression (Ridge and Lasso regularization)
- XGBoost (optimized with n_estimators=100, max_depth=6)
- CatBoost (iterations=1000, depth=6)

## Results

| Model | Focused F1 | Unfocused F1 | Drowsy F1 | Avg F1 | Accuracy |
|-------|------------|--------------|-----------|---------|----------|
| **XGBoost** | **0.80** | **0.77** | **0.67** | **0.75** | **0.74** |
| CatBoost | 0.80 | 0.76 | 0.65 | 0.74 | 0.74 |
| RLDA | 0.79 | 0.75 | 0.63 | 0.72 | 0.72 |
| Logistic Regression (Ridge) | 0.70 | 0.67 | 0.52 | 0.63 | 0.63 |
| Random Forest | 0.71 | 0.64 | 0.50 | 0.62 | 0.62 |
| Logistic Regression (Lasso) | 0.77 | 0.70 | 0.18 | 0.55 | 0.63 |

The **XGBoost model** achieved the best overall performance with 74% accuracy and 0.75 average F1-score, significantly improving drowsy state detection from 0.51 to 0.67 F1-score (31% improvement).

## Discussion

- **Focused vs. Unfocused States**: All models performed well in distinguishing these states, with tree-based ensemble methods achieving F1-scores above 0.75
- **Drowsy State Detection**: This remained the most challenging class, though XGBoost and CatBoost showed significant improvements over traditional methods
- **Feature Engineering Impact**: The combination of band-pass filtering and PCA proved crucial for extracting meaningful patterns while reducing computational complexity

## Limitations and Future Work

### Current Limitations:
- Limited to 23 subjects, which may affect model generalizability
- No temporal modeling of sequential EEG patterns
- Single train-test split evaluation (future work should include cross-validation)

### Future Directions:
- Implement deep learning approaches (CNNs, LSTMs) to capture temporal dependencies
- Explore real-time implementation for practical BCI applications
- Investigate subject-specific model adaptation
- Extend to multi-modal data fusion (EEG + eye tracking)

## References

This project was inspired by:
- Inan, C., Kaya, M., & Mishchenko, Y. (2019). Distinguishing mental attention states of humans via an EEG-based passive BCI using machine learning methods. *Expert Systems with Applications*, 134, 153–166.

## Repository Structure
```
├── README.md
├── eeg-attention-state-classifier.ipynb  # Main implementation notebook
├── requirements.txt                      # Python dependencies
└── results/                             # Model performance visualizations
```

## Requirements
- Python 3.10+
- NumPy, Pandas, Scikit-learn
- XGBoost, CatBoost
- MNE-Python (for EEG processing)
- Matplotlib, Seaborn (for visualization)
