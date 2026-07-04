# Bird Audio Edge AI

PyTorch-based bird species classification using Mel spectrograms and convolutional neural networks (CNNs), featuring audio preprocessing, noise handling, and Edge AI deployment considerations for TinyML research.

## Project Overview

This project implements an end-to-end bioacoustic classification pipeline for recognizing bird species from audio recordings. Raw audio data are preprocessed into Mel spectrogram representations and fed into a lightweight CNN model for classification.

The project explores practical challenges in real-world Edge AI applications, including:

- Variable-length audio recordings
- Stereo and mono audio conversion
- Background noise and silent intervals
- Fixed-duration normalization
- Mel spectrogram feature extraction
- Lightweight CNN architectures suitable for future TinyML and FPGA acceleration

The current implementation serves as a foundation for future research in Edge AI, embedded machine learning, and AI accelerator systems.

---

## Dataset

The dataset contains recordings from five bird species:

- American Robin
- Blue Jay
- House Sparrow
- Mourning Dove
- Northern Cardinal

The raw audio files are provided in `audio.zip`.

Current dataset statistics:

- Total samples: 70
- Training samples: 56
- Testing samples: 14

---

## Audio Preprocessing Pipeline

The preprocessing pipeline includes several steps to improve training quality:

```text
Raw Audio
    ↓
Mono Conversion
    ↓
Resampling (32 kHz)
    ↓
Waveform Normalization
    ↓
Silence Trimming
    ↓
Fixed-Duration Padding/Cropping
    ↓
Mel Spectrogram Generation
    ↓
CNN Classification
