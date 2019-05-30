import os
import scipy.io
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CarsDataset(Dataset):
    """Custom data set"""

    def __init__(self, mat_anno, data_dir, car_names, transform=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.full_data_set = scipy.io.loadmat(mat_anno)

        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):

        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name)

        label = self.car_annotations[idx][-2][0][0]

        bbox = (self.car_annotations[idx][-6][0][0],
                self.car_annotations[idx][-5][0][0],
                self.car_annotations[idx][-4][0][0],
                self.car_annotations[idx][-3][0][0])

        bbox = torch.from_numpy(np.array(bbox, dtype=np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label, bbox
