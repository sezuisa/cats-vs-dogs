# Cats vs. Dogs - a little CNN project

By Sarah Hägele & Sandro Bühler

*For the basic model, we followed [this](https://www.geeksforgeeks.org/cat-dog-classification-using-convolutional-neural-network-in-python/) tutorial by GeeksForGeeks.*

---

## Models
For all models, the [Asirra Dataset](https://www.kaggle.com/c/dogs-vs-cats/data) was split into a subset of 22.500 training images and 2.500 validation images.

| Model | Epochs | Architecture | Training Accuracy | Validation Accuracy |
|---|---|---|---|---|
| model_1 | 10 | 4 Conv. Layers (filters: 32, 64, 64, 64; kernel_size: 3x3, activation: relu) with MaxPooling; Flatten Layer; 3 Dense Layers (units: 512, activation: relu) with BatchNormalization & Dropout (rate: 0.1, 0.2), Output Layer (activation: sigmoid) | 89,44% | 83,00% |
|   |   |   |   |   |
|   |   |   |   |   |

## Thoughts and Remarks

Some remarks.