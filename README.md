# DivGI
"DivGI: Delve into Digestive Endoscopy Image Classification" \
This repo is modified version of [DivGI](https://github.com/howardchina/DivGI). Modification includes training and testing scripts for single label classification.

## Features
- Multi-class classification (default: 23 classes)
- Balanced MixUp augmentation for better generalization
- Multi-granular training option
- Training/validation loop with:
  - Accuracy, Precision, Recall, F1, Balanced Accuracy
  - Learning rate scheduling
  - CSV logging of metrics
- Automatic saving of best model (`best_divgl.pth`)


## ⚙️ Requirements
- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- tqdm
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure
```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```
## Training
```
python train.py
```
