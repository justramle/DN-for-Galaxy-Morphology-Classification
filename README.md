# Enhanced Comparative Analysis of Pretrained and Custom Deep CNNs for Galaxy Morphology Classification

A comparison of three convolutional neural network approaches: a CNN built from scratch, a ResNet-50, and an EfficientNet-B5: on the task of classifying galaxy morphology from the [Galaxy10 DECals](https://astronn.readthedocs.io/en/latest/galaxy10.html) dataset. This project received a **Best Paper Award** (see [`certificate/`](certificate)).


## Dataset

[Galaxy10 DECals](https://astronn.readthedocs.io/en/latest/galaxy10.html) (loaded via [`astroNN`](https://github.com/henrysky/astroNN)): ~17,700 labeled 256×256 RGB galaxy images across 10 morphology classes.

| Class | Description |
|---|---|
| 0 | Distributed Galaxies |
| 1 | Merging Galaxies |
| 2 | Round Smooth Galaxies |
| 3 | In-between Round Smooth Galaxies |
| 4 | Cigar Shaped Smooth Galaxies |
| 5 | Barred Spiral Galaxies |
| 6 | Unbarred Tight Spiral Galaxies |
| 7 | Unbarred Loose Spiral Galaxies |
| 8 | Edge-on Galaxies without Bulge |
| 9 | Edge-on Galaxies with Bulge |

Class 4 is heavily underrepresented, so the custom CNN pipeline oversamples it 3x with augmentation (rotation, shift, zoom, flip) before training.


## Models

| Model | Script | Approach |
|---|---|---|
| Custom CNN | [`Galaxy_Morphological_CustomModel.py`](Galaxy_Morphological_CustomModel.py) | 4-conv-block CNN trained from scratch (Keras/TensorFlow), 128×128 input |
| ResNet-50 | [`Galaxy_Morphological_RestNet50.py`](Galaxy_Morphological_RestNet50.py) | ResNet-50 backbone + custom FC head (PyTorch), 224×224 input |
| EfficientNet-B5 | [`Galaxy_Morphological_EfficientNetB5_2.py`](Galaxy_Morphological_EfficientNetB5_2.py) | ImageNet-pretrained EfficientNet-B5 (`timm`) + custom FC head (PyTorch), 224×224 input |

All three use early stopping on validation accuracy/loss, and the PyTorch models train with mixed precision (`torch.cuda.amp`).

## Results

| Model | Accuracy | Notes |
|---|---|---|
| Custom CNN | 69.4% (test) / 67.1% (val) | Trained from scratch, no pretrained weights |
| ResNet-50 | 80.4% | Randomly initialized backbone |
| EfficientNet-B5 | 87.0% | ImageNet-pretrained backbone, best performer |


## Setup

```bash
conda create -n galaxy-cnn python=3.10
conda activate galaxy-cnn
pip install astroNN tensorflow torch torchvision timm scikit-learn opencv-python pillow matplotlib seaborn tqdm
```

## Usage

Each script downloads the Galaxy10 dataset on first run (~2.7GB) and trains end to end, saving the best checkpoint to `checkpoints/` and result plots to the working directory.

```bash
python Galaxy_Morphological_CustomModel.py
python Galaxy_Morphological_RestNet50.py
python Galaxy_Morphological_EfficientNetB5_2.py
```
