import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import torch.quantization
ImageFile.LOAD_TRUNCATED_IMAGES = True

class RawDataset:
    def __init__(self):
        self.train_set = "/Users/pyuvraj/Downloads/nudity_datasets/nude_sexy_safe_v1_x320/training"
        self.test_set = "/Users/pyuvraj/Downloads/nudity_datasets/nude_sexy_safe_v1_x320/testing"
        self.val_set = "/Users/pyuvraj/Downloads/nudity_datasets/nude_sexy_safe_v1_x320/validation"

    def prepareTrainingSet(
        self, set_location="/home/pyuvraj/CCPP/nudity-pytorch/data/P2datasetFull/train"
    ):
        positive_label = set_location + "/nude/"
        negative_label = set_location + "/safe/"
        # training_positive_label = os.listdir(positive_label)
        # training_negative_label = os.listdir(negative_label)
        training_positive_label = [file for file in os.listdir(positive_label) if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')]
        training_negative_label = [file for file in os.listdir(negative_label) if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')]
        training_positive_dataset = []
        training_negative_dataset = []
        for i in range(1, len(training_positive_label)):
            training_positive_dataset.append(
                [positive_label + training_positive_label[i], True]
            )
        for i in range(1, len(training_negative_label)):
            training_negative_dataset.append(
                [negative_label + training_negative_label[i], False]
            )
        print("number of positive samples: ", len(training_positive_dataset))
        print("number of negative samples: ", len(training_negative_dataset))
        return training_positive_dataset, training_negative_dataset

class AdultContentDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image_path = self.data[idx][0]
        image = Image.open(image_path).convert('RGB')
        label = 1 if self.data[idx][1] is True else 0
        if self.transform:
            image = self.transform(image)
        return image, label


torch.manual_seed(42)
transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = RawDataset()
train_pos, train_neg = dataset.prepareTrainingSet(dataset.train_set)
val_pos, val_neg = dataset.prepareTrainingSet(dataset.val_set)
test_pos, test_neg = dataset.prepareTrainingSet(dataset.test_set)
# print(train_pos[1:10])


# Create the dataset and dataloader
scrapped_training_dataset = train_pos + train_neg
random.shuffle(scrapped_training_dataset)
training_dataset = AdultContentDataset(scrapped_training_dataset, transform=transform)
train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)

scrapped_validation_dataset = val_pos + val_neg
random.shuffle(scrapped_validation_dataset)
val_dataset = AdultContentDataset(scrapped_validation_dataset, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

scrapped_test_dataset = test_pos + test_neg
random.shuffle(scrapped_test_dataset)
test_dataset = AdultContentDataset(scrapped_training_dataset, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
