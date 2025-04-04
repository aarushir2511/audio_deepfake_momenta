# AI-Generated Speech Detection Using CNNs

This repository contains a Convolutional Neural Network (CNN)-based system for detecting AI-generated speech. The model processes audio files by converting them into Mel-spectrograms and classifies them as either **bonafide (human)** or **spoof (AI-generated)**.

## Dataset

- **Source**: [ASVspoof 2019 Logical Access Subset](https://www.asvspoof.org/index2019.html)
- Format: `.flac` audio files
- Labels: `bonafide` (genuine speech) and `spoof` (AI-synthesized speech)
- A partial subset was used due to storage constraints, ensuring a representative mix of synthesis systems

## Model Overview

- **Architecture**: Shallow CNN with two convolutional blocks followed by dense layers
- **Input**: Mel-spectrograms (128 frequency bins × 109 time frames)
- **Preprocessing**: 
  - Resampling to 16 kHz
  - 5-second clip duration (truncated or padded)
  - Mel-spectrogram extraction with `librosa`
  - Conversion to decibel scale
- **Output**: Binary classification (bonafide/spoof) using softmax

## Performance

Evaluated on the ASVspoof 2019 LA subset:

- **Accuracy**: ~92%
- **Precision**: ~90%
- **Recall**: ~94%
- **F1 Score**: ~92%
- **AUC (ROC)**: 0.95

## Achievements

- Achieved strong classification performance on a real-world benchmark dataset
- Efficient CNN model suitable for deployment and real-time inference
- Robust preprocessing pipeline using open-source tools like `librosa` and `numpy`
- Visualizable and interpretable intermediate CNN outputs

## Getting Started

To clone the repository:

```bash
git clone https://github.com/yourusername/ai-speech-detection.git
cd ai-speech-detection
