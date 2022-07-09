from typing import Tuple
import torch
import os
import cv2
from torch.utils.data import DataLoader
from digit_recognition.data import utils

class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform=None) -> None:
        super().__init__()

        self.images, self.labels = self._get_images(dataset_path)
        self.trasform = transform

    def _get_images(self, folder: str, img_extensions: Tuple=(".jpeg", ".jpg", ".png")):
        images= []
        labels = []
        for root, _, files in os.walk(folder):
            for img in files:
                if img.endswith(img_extensions):
                    img_path = os.path.abspath(os.path.join(root, img))
                    images.append(img_path)
                    label = os.path.split(root)[-1]
                    labels.append(int(label))
        return images, labels

    def __len__(self, ):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        image = torch.permute(image, (2, 0, 1)).float()

        label = self.labels[idx] - 1
        if self.trasform:
            image = self.trasform(image).float()
        
        return image, label
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        default="/Users/mo/Projects/Digit_Recognition_Old/Data/All Digits/training")
    parser.add_argument("--transforms", type=str, default='train')
    args = parser.parse_args()

    if args.transforms in ['validation', 'testing']:
        transform = utils.val_data_transforms
    else:
        transform = utils.data_transforms

    dataset = DigitDataset(args.dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, num_workers=8, shuffle=True)
    utils.visualize_dataloader(dataloader, rows = 3, columns = 4)


