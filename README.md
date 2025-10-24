# Enhancing ECG Arrhythmia Detection Robustness through Noise-Augmented Training

A deep learning approach for robust ECG arrhythmia classification that maintains high accuracy even in noisy, real-world clinical conditions.

## Overview

This project develops a noise-robust Convolutional Neural Network (CNN) for automated arrhythmia detection from ECG signals. Traditional deep learning models trained on clean data often fail when faced with real-world noise from muscle artifacts, baseline wander, and electronic interference. This project addresses this challenge through **on-the-fly noise augmentation** during training, improving classification accuracy by up to 18% in noisy conditions.

## Key Features

- **Noise-Augmented Training**: Random injection of Additive White Gaussian Noise (AWGN) at various SNR levels during training
- **Robust Performance**: 8.5% average accuracy improvement over baseline CNN across all noise levels
- **1D-CNN Architecture**: Hierarchical feature extraction optimized for temporal ECG signals
- **Real-World Ready**: Tested across 8 SNR levels (0-20 dB) mimicking clinical monitoring scenarios
- **No Preprocessing Required**: Works directly with raw ECG signals without complex denoising pipelines

## Problem Statement

Cardiovascular diseases cause 31% of global deaths, with arrhythmias being particularly critical. While deep learning models achieve high accuracy on clean ECG data, they suffer significant performance degradation when exposed to real-world noise from wearable devices and ambulatory monitoring. This gap between laboratory performance and clinical applicability limits their deployment in healthcare settings.

## Solution

The proposed CNN model is trained with dynamic noise augmentation where:
- 70% of training samples receive random AWGN at SNR levels between 0-20 dB
- 30% remain clean to preserve performance on high-quality signals
- The model learns noise-invariant features, making it robust to various real-world conditions

## Dataset

**MIT-BIH Arrhythmia Database**
- 100,033 labeled heartbeats from 48 half-hour recordings
- Sampled at 360 Hz with expert annotations
- 216-dimensional feature vectors centered on R-peaks
- 5 arrhythmia classes: Normal (N), Atrial Premature (A), Left Bundle Branch Block (L), Right Bundle Branch Block (R), Premature Ventricular Contraction (V)

## Model Architecture

**1D Convolutional Neural Network**:
- 3 convolutional blocks with increasing filters (32, 64, 128)
- Batch normalization and ReLU activation after each block
- Max-pooling for dimension reduction
- 2 fully-connected layers with dropout for regularization
- Softmax output layer for 5-class classification

## Results

| SNR (dB) | Baseline CNN | Noise-Augmented CNN | Improvement |
|----------|--------------|---------------------|-------------|
| 0 | 71.23% | 88.54% | +17.31% |
| 3 | 76.78% | 95.09% | +18.31% |
| 6 | 84.93% | 98.18% | +13.25% |
| 9 | 90.38% | 99.21% | +8.83% |
| 12 | 94.29% | 99.57% | +5.28% |
| 15 | 96.86% | 99.67% | +2.81% |
| 18 | 98.34% | 99.74% | +1.40% |
| 20 | 98.91% | 99.76% | +0.85% |
| **Average** | **88.97%** | **97.47%** | **+8.50%** |

### Key Findings

- **Best validation accuracy**: 92.9%
- **Maximum improvement**: 18.31% at 3 dB SNR
- **Consistent gains**: Performance improvements across all noise levels
- **Maintains clean signal performance**: No degradation on high-quality data

## Training Details

- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 64
- **Epochs**: 30
- **Noise Augmentation Probability**: 70%
- **SNR Range**: 0, 3, 6, 9, 12, 15, 18, 20 dB

