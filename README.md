# Cats vs. Dogs - a little CNN project

By Sarah Hägele & Sandro Bühler

*For the basic model, we followed [this](https://www.geeksforgeeks.org/cat-dog-classification-using-convolutional-neural-network-in-python/) tutorial by GeeksForGeeks.*

---

## Models
For all models, the [Asirra Dataset](https://www.kaggle.com/c/dogs-vs-cats/data) was split of 80% for training and 20% for validation.

| Model | Epochs | Architecture | Training Accuracy | Validation Accuracy | Time |
|---|---|---|---|---|---|
| model_1 | 10 | VGG-1 | 98,41% | 70,216% | 00:15:09 |
| model_1-2 | 10 | VGG-1 with `image_dataset_from_directory` | 91,52% | 68,22% | 00:15:54 |
| model_2 | 10 | VGG-2 | 99,64% | 76,09% | 00:15:22 |
| model_3 | 10 | VGG-3 | 98,49% | 78,978% | 00:18:22 |
| model_3-2 | 10 | VGG-3 with Dropout | 88,52% | 82,633% | 00:17:50 |
| model_3-3 | 10 | VGG-3 with Image Data Augmentation | 86,42% | 85,914% | 00:20:21 |
| model_3-4 | 10 | VGG-3 with Dropout and Image Data Augmentation | 81,99% | 84,597% | 00:21:17 |
| model_3-4 | 50 | VGG-3 with Dropout and Image Data Augmentation | 91,09% | 91,513% | 01:46:30 |

Hardware used for training these models:

Apple MacBook Air | Apple M1 Chip | 16 GB RAM | 8 CPU Cores | 8 GPU Cores

## Thoughts and Remarks

Some remarks.