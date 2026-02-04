# Swahili Gemma 3 1B Continued Pretraining - NIRU AI Hackathon

Continued pretraining and finetuning of Swahili Gemma 3 1B model on a custom 18M Swahili linguistic corpus.

## Project Overview

Adapting Gemma 3 1B for improved Swahili language understanding and conversation through continued pretraining and finetuning:
- WAXAL Swahili TTS dataset (Makerere University)
- Swahili educational content
- Custom curated Swahili corpus

##  Dataset

### Sources
1. **WAXAL TTS** (CC-BY-SA-4.0)
2. **Swahili Linguistic Content**
3. **Custom Corpus**

**Estimated Total:** 50-200M tokens

##  Quick Start

### Setup
```powershell
# Install dependencies
pip install -r requirements.txt
```

### Download Data
```powershell
python scripts/01_download_waxal.py
```

### Preprocess
```powershell
python scripts/02_preprocess.py
```

### Train
```powershell
python scripts/03_train.py
```

## 📁 Structure
```
├── data/              # Datasets (not in git)
├── scripts/           # Python scripts
├── notebooks/         # Jupyter notebooks
├── configs/           # Config files
├── models/            # Checkpoints (not in git)
└── results/           # Metrics & plots
```

## 📈 Progress

- [x] Project setup
- [x] Data collection
- [ ] Preprocessing
- [ ] Training
- [ ] Evaluation

##  Hackathon

**Event:** NIRU Hackathon 2026  
**Track:** Generative  AI  
**Date:** February 2026

## 📝 License

- Code: MIT
- WAXAL Data: CC-BY-SA-4.0
- Gemma Model: Gemma Terms of Use