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
    def __init__(self, root="/CCPP/nudity-pytorch/data/P2datasetFull/"):
        self.train_set = "/Users/pyuvraj/Downloads/Nudity dataset/P2datasetFull/train"
        self.test_set = "/Users/pyuvraj/Downloads/Nudity dataset/P2datasetFull/test1"
        self.val_set = "/Users/pyuvraj/Downloads/Nudity dataset/P2datasetFull/val1"
        self.final_results = "/Users/pyuvraj/Downloads/Nudity dataset/imageframes"

    def prepareTrainingSet(
        self, set_location="/home/pyuvraj/CCPP/nudity-pytorch/data/P2datasetFull/train"
    ):
        positive_label = set_location + "/2/"
        negative_label = set_location + "/1/"
        training_positive_label = os.listdir(positive_label)
        training_negative_label = os.listdir(negative_label)
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
        return training_positive_dataset, training_negative_dataset

    def prepareFinalTestSet(
        self, set_location="/home/pyuvraj/CCPP/data/imageframes/"
    ):
        training_positive_label = os.listdir(set_location)
        final_test_list = []
        for i in range(1, len(training_positive_label)):
            final_test_list.append(
                [set_location + "/" + training_positive_label[i], True]
            )
        return final_test_list


if __name__ == "__main__":
    dataset = RawDataset()
    train_pos, train_neg = dataset.prepareTrainingSet(dataset.train_set)
    val_pos, val_neg = dataset.prepareTrainingSet(dataset.val_set)
    test_pos, test_neg = dataset.prepareTrainingSet(dataset.test_set)
    final_test = dataset.prepareFinalTestSet(dataset.final_results)
    print("done")
