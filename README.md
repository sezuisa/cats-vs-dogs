# Cats vs. Dogs - a little CNN project

By Sarah Hägele & Sandro Bühler

*For the basic model, we followed [this](https://www.geeksforgeeks.org/cat-dog-classification-using-convolutional-neural-network-in-python/) tutorial by GeeksForGeeks.*

---

## Models
For all models, the [Asirra Dataset](https://www.kaggle.com/c/dogs-vs-cats/data) was split of 80% for training and 20% for validation.

| Model | Epochs | Architecture | Training Accuracy | Validation Accuracy | Time | Hardware |
|---|---|---|---|---|---|---|
| model_1 | 10 | VGG-1 | 98,41% | 70,216% | 00:15:09 | Sarah Hägele |
| model_1-2 | 10 | VGG-1 with `image_dataset_from_directory` | 91,52% | 68,22% | 00:15:54 | Sarah Hägele |
| model_2 | 10 | VGG-2 | 99,64% | 76,09% | 00:15:22 | Sarah Hägele |
| model_3 | 10 | VGG-3 | 98,49% | 78,978% | 00:18:22 | Sarah Hägele |
| model_3-2 | 10 | VGG-3 with Dropout | 88,52% | 82,633% | 00:17:50 | Sarah Hägele |
| model_3-3 | 10 | VGG-3 with Image Data Augmentation | 86,42% | 85,914% | 00:20:21 | Sarah Hägele |
| model_3-4 | 10 | VGG-3 with Dropout and Image Data Augmentation | 81,99% | 84,597% | 00:21:17 | Sarah Hägele |
| model_3-4 | 50 | VGG-3 with Dropout and Image Data Augmentation | 91,09% | 91,513% | 01:46:30 | Sarah Hägele |
| model_3-4 | 100 | VGG-3 with Dropout and Image Data Augmentation | 92,77% | 93,46% | 17:23:05 | Sandro Bühler (CPU only) |
| model_3-4 | 100 | VGG-3 with Dropout and Image Data Augmentation | 92,96% | 93,95% | 5:57:47 | Sandro Bühler (CPU + GPU)|

Hardware used for training these models:
Sarah Hägele:
Apple MacBook Air | Apple M1 Chip | 16 GB RAM | 8 CPU Cores | 8 GPU Cores
Sandro Bühler:
i5-7600K 5,1 GHz | 4 CPU Cores | 32 GB RAM | GTX 1080 | 8 GB VRAM

## Thoughts and Remarks

Some remarks.