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
from data_ingestion_pipeline_smaller_dataset import RawDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

# Define the transforms to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
torch.manual_seed(42)

dataset = RawDataset()
train_pos, train_neg = dataset.prepareTrainingSet(dataset.train_set)
val_pos, val_neg = dataset.prepareTrainingSet(dataset.val_set)
test_pos, test_neg = dataset.prepareTrainingSet(dataset.test_set)
final_test = dataset.prepareFinalTestSet(dataset.final_results)
print("done")
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

final_test_dataset = AdultContentDataset(final_test, transform=transform)
final_test_dataloader = DataLoader(final_test_dataset, batch_size=1, shuffle=False)
