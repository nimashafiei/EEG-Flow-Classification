# EEG-Flow-Classification

EEG Flow State Classification using entropy-based features and deep learning models (including a custom FT-Transformer).  
This project applies machine learning and deep learning to classify EEG signals into **activity (game)** and **rest** states, aiming to detect flow-related patterns in neural activity.

---

## 🚀 Features
- Modular **OOP Python design** (separated into `data/`, `models/`, `training/`, `utils/`)  
- **Entropy feature extraction** (`XSpecEn`, `XDistEn`, etc.)  
- **Rest-centering preprocessing** for subject-specific normalization  
- Classical ML models (e.g., Random Forests) + **custom FT-Transformer (PyTorch)**  
- **Weighted soft-voting ensemble** across entropies  
- **Leave-One-Subject-Out (LOSO) evaluation** for subject-independent performance  

---

## 📂 Project Structure


EEG-Flow-Classification/
│── config.py # Global experiment configuration
│── main.py # Entry point (training & evaluation)
│
├── data/
│   └── loader.py # Load entropy CSVs per subject
│
├── models/
│   └── ft_transformer.py # Custom FT-Transformer model
│
├── training/
│   ├── trainer.py # Training loop & evaluation
│   └── evaluation.py # Metrics & summaries
│
└── utils/
    └── preprocess.py # Rest-centering & preprocessing

---

## ⚡ Getting Started

### 1. Clone repository
```bash
git clone https://github.com/nimashafiei/EEG-Flow-Classification.git
cd EEG-Flow-Classification
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

Dependencies include:
- torch
- scikit-learn
- pandas
- numpy
- imbalanced-learn
- matplotlib

### 3. Run training (example)
```bash
python main.py
```

By default, main.py runs FT-Transformer training on one entropy with a leave-one-subject-out setup.

---

## 📊 Results
- Robust classification of EEG signals into game vs. rest  
- Subject-independent performance via LOSO cross-validation  
- Weighted soft-voting improves ensemble accuracy and macro-F1  

---

## 🙌 Contributions
Pull requests and suggestions are welcome.

---

## 📜 License
MIT License
