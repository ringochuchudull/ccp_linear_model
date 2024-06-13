# _AUTHOR_ Ringo S.W Chu

import os

import glob
from PIL import Image

import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class VesselImageDataset(Dataset):

    def __init__(self, image_classes, transform=None):

        self.folders = image_classes

        self.labels = []
        self.imgpath = []

        for idx, cls in enumerate(self.folders):
            temp =  glob.glob(os.path.join('data', cls, '*'))   

            self.imgpath += temp
            self.labels += [idx] *  len(temp)

        assert len(self.imgpath) == len(self.labels)
        self.num_classes = len(self.folders)

        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                                    transforms.Resize((128, 128)),  # Resize images to a fixed size
                                    transforms.ToTensor(),  # Convert images to PyTorch tensors
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.targets_transform = None
        if self.targets_transform is None:
            self.targets_transform = transforms.Compose([
                                        transforms.ToTensor()])

    def __getitem__(self, idx):
        img = Image.open(self.imgpath[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img, label

    def __len__(self):
        return len(self.imgpath)

if __name__ == '__main__':

    # Test the dataloader
    image_classes = ["OOCL_VESSEL_SHIPS", "CARGO_TRUCKS"]
    dataset = VesselImageDataset(image_classes)
    img1, label1 = dataset[0]

