# Facial Emotion Recognition with CNN — FER-2013

A Convolutional Neural Network trained to classify human facial expressions into 7 emotion categories, framed as an extension of Sturm et al.'s (2022) *awe walks* study on emotional well-being in older adults.

> **Course:** Data Mining (Minería de Datos) — Facultad de Ingeniería, UNAM  
> **Professor:** M.C. Jose Roberto Olvera Lopez  
> **Team:** Equipo 10

---

## Background

Sturm et al. (2022) conducted a randomized controlled trial with 60 older adults (ages 60–90) to test whether *awe walks* improve emotional well-being. Participants took selfies during weekly walks; the authors measured behavioral change via MATLAB-computed *self-size* and manually coded smile intensity.

This project proposes automating that analysis pipeline: replacing manual smile coding with a CNN classifier that maps facial expressions to 7 discrete emotions, enabling longitudinal emotional trajectory tracking at scale.

Since the original study's selfies are not publicly available (privacy/ethics restrictions), the model is trained on **FER-2013** as a proxy dataset. The architecture can be fine-tuned on the study's corpus once access is granted.

## Results

| Emotion  | Precision | Recall | F1   | Support |
|----------|-----------|--------|------|---------|
| Angry    | 0.52      | 0.58   | 0.55 | 958     |
| Disgust  | 0.80      | 0.18   | 0.29 | 111     |
| Fear     | 0.49      | 0.32   | 0.38 | 1024    |
| **Happy**| **0.85**  | **0.86**| **0.85** | 1774 |
| Neutral  | 0.56      | 0.69   | 0.62 | 1233    |
| Sad      | 0.49      | 0.50   | 0.50 | 1247    |
| Surprise | 0.76      | 0.77   | 0.76 | 831     |
| **Overall accuracy** | | | **0.63** | 7178 |

`Happy` (the primary emotion of interest for the awe walks study) achieves F1 = 0.85, validating the classifier for the intended use case.

## Architecture

```
Conv2D(32) → BN → MaxPool
Conv2D(64) → BN → MaxPool
Conv2D(128) → BN → MaxPool
Flatten → Dense(256) + Dropout(0.5) → Dense(7, softmax)
```

- **Input:** 48×48 grayscale images  
- **Trainable parameters:** 619,463  
- **Optimizer:** Adam | **Loss:** categorical crossentropy  
- **Training callbacks:** ModelCheckpoint, ReduceLROnPlateau, EarlyStopping  
- **Hardware:** NVIDIA GeForce RTX 3060 (CUDA 12)

## Repository Structure

```
├── Practica 3.6 Redes Neuronales Equipo 10.ipynb   # Spanish notebook (original)
├── CNN_FER2013_Team10_EN.ipynb                      # English notebook
├── doc/
│   ├── Practica_3.6_Redes_Neuronales_Equipo_10.tex # Spanish LaTeX report
│   ├── Practica_3.6_Redes_Neuronales_Equipo_10.pdf # Spanish PDF report
│   ├── CNN_FER2013_Team10_EN.tex                   # English LaTeX report
│   ├── CNN_FER2013_Team10_EN.pdf                   # English PDF report
│   └── img/
│       ├── accuracy-and-loss.png
│       └── confusion-matrix.png
└── data/                                            # FER-2013 dataset (not tracked)
```

## Setup

**Requirements:** Python 3.11, CUDA 12-compatible GPU (optional but recommended)

```bash
# Clone and create virtual environment
git clone <repo-url>
cd <repo>
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install "tensorflow[and-cuda]" ipykernel matplotlib scikit-learn seaborn pandas

# Register Jupyter kernel
python -m ipykernel install --user --name p6-mineria --display-name "P6 Mineria (GPU)"

# Download dataset from Kaggle
# Place at data/fer2013/train/ and data/fer2013/test/
```

**Dataset:** [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) — download manually and extract to `data/fer2013/`.

## Reference

Sturm, V. E., et al. (2022). *Big smile, small self: Awe walks promote prosocial positive emotions in older adults.* Emotion, 22(5), 1044–1058. https://doi.org/10.1037/emo0000876
