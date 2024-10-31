
**README.md**


# DeepVision

DeepVision is a Convolutional Neural Network (CNN) designed for the classification of grayscale images into 20 distinct categories. This repository implements a deep architecture with batch normalization, dropout, and advanced data augmentation, yielding a robust model that has reached over 76% accuracy on a dataset of 130,000 images.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Model Architecture](#model-architecture)
- [Data Augmentation Techniques](#data-augmentation-techniques)
- [Training Details](#training-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
DeepVision is built for grayscale image classification using CNNs. The model leverages advanced augmentations and a deep architecture optimized with techniques like batch normalization and dropout. The project also includes various data preprocessing and augmentation scripts to ensure the model generalizes well across different data distributions.

## Features
- **Deep CNN Architecture** with multiple convolutional layers, max pooling, and dense layers.
- **Batch Normalization and Dropout** to improve generalization and prevent overfitting.
- **Data Augmentation** for robust training on diverse data variations.
- **Training and Validation Scripts** to evaluate the model’s performance.

## Installation
To clone and run this project, ensure you have Python 3.7+ installed, along with the necessary libraries in `requirements.txt`:
```bash
git clone https://github.com/Aleksandar-Mladenoski/DeepVision.git
cd DeepVision
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**:
   - Place your dataset in the `data/` directory, structured into train/test folders.
   - Run `preload_dataset.py` to preprocess the data for training.

2. **Data Augmentation**:
   - Use `augmentdata.py` and `complex_augment.py` to generate augmented datasets.
   - Customize augmentations in these scripts as needed.

3. **Training**:
   - Run `train.py` or `training_main.py` to start the training process.
   - Adjust hyperparameters within these files or configure them via command-line arguments.

4. **Evaluation**:
   - Evaluate the model’s performance using `test_model_validation.py`.

## File Descriptions

| File                     | Description                                                      |
|--------------------------|------------------------------------------------------------------|
| `architecture.py`        | Defines the CNN model architecture and layers.                  |
| `augmentdata.py`         | Standard data augmentation script.                              |
| `complex_augment.py`     | Implements more complex augmentations.                          |
| `data.py`                | Manages dataset loading and pre-processing functions.           |
| `dataset.py`             | Custom dataset classes to handle data pipelines.               |
| `precompdata.py`         | Pre-compute certain data features to save processing time.      |
| `preload_dataset.py`     | Initial data loading and basic preprocessing script.            |
| `train.py`               | Script to train the CNN model.                                  |
| `training_main.py`       | Main training loop with custom training configurations.         |
| `test_model_validation.py`| Evaluates model performance on validation/test sets.         |
| `save_augment_data.py`   | Saves augmented data for repeated use without reprocessing.     |

## Model Architecture
The CNN consists of multiple convolutional layers with the following characteristics:
- **Input**: 1-channel grayscale images, resized to 100x100 pixels.
- **Hidden Layers**: Feature extraction with [5x5 kernels, increasing depth].
- **Pooling**: Max pooling applied every two layers to reduce spatial dimensions.
- **Fully Connected Layers**: Feedforward layers with dropout for regularization.
- **Output**: Softmax activation for multi-class classification (20 classes).

## Data Augmentation Techniques
Data augmentation strategies include:
- **Standard Augmentation**: Rotation, flipping, scaling, and cropping.
- **Complex Augmentation**: Introduces brightness, contrast, noise addition, and advanced distortions for robust training.

## Training Details
To train the model, run:
```bash
python train.py
```
Or for custom configurations:
```bash
python training_main.py --epochs 20 --batch_size 64 --learning_rate 0.001
```

## Results
The model achieved over 76% accuracy at epoch 14 on a test set, with steady improvement observed across 20 epochs.

## Contributing
If you wish to contribute to DeepVision, feel free to fork the repository, make your changes, and submit a pull request. Please ensure any contributions are tested.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
```
---
