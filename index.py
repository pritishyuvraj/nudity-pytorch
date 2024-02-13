from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch, torchvision
import os
import matplotlib.pyplot as plt
import numpy as np

class NudityDetection:
    def __init__(self):
        pass

    def show(self, img):
        print(img.shape)
        plt.imshow(img)
        print("printed")
        # plt.imshow(  img.permute(1, 2, 0)  )
        # npimg = img.numpy()
        # plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

    def custom_collate(self,batch):
        return torch.utils.data.dataloader.default_collate(list(map(lambda x: torchvision.transforms.functional.to_tensor(x), batch)))

    def downloadBoudoirDataset(self, output_dir="boudoir_dataset"):
        # boudoir Dataset
        dataset = load_dataset("soymia/boudoir-dataset")
        torch_dataset = dataset.with_format("torch")
        print("Torch dataset -> ", torch_dataset, torch_dataset["train"][0]['image'])
        self.show(torch_dataset["train"][0]['image'])
        # dataset.save_to_disk("./data")
        # print(dataset)
        # print(dataset["train"][0])
        # print(type(dataset["train"][0]['image']), dir(dataset["train"][0]['image']))
        # print("file name", dataset["train"][0]['image'].filename)

if __name__ == '__main__':
    nudityDetection = NudityDetection()
    nudityDetection.downloadBoudoirDataset()
