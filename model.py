import logging
import warnings
import torch
import time
import torch.nn.functional as F
from torch import nn
from torchvision import models
import torch.optim as optim
from predata import CarsDataset


warnings.filterwarnings("ignore")
module_logger = logging.getLogger("main.model")


class DenseNet161(nn.Module):

    def __init__(self, num_classes=196, drop_rate=0):
        super(DenseNet161, self).__init__()

        # create base model
        original_model = models.densenet161(pretrained=True, drop_rate=drop_rate)
        # get all layers from original model without last layer
        # and create our custom layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])

        # TODO *4?
        self.out_labels = (nn.Linear(2208, num_classes))
        self.out_bbox = nn.Linear(2208, 4)

    def forward(self, x):
        out = self.features(x)
        out = F.relu(out, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(out.size(0), -1)

        x_labels = self.out_labels(out)
        x_bbox = self.out_bbox(out)

        return x_labels, x_bbox

    def save_model(self):
        pass

    def load_model(self):
        pass

    @staticmethod
    def train_model():
        logger = logging.getLogger('main.model.Densenet161.train')

        mat_anno = '/home/mirage/DNN/datasets/devkit/cars_train_annos.mat'
        data_dir = '/home/mirage/DNN/datasets/cars_train'
        car_names = '/home/mirage/DNN/datasets/devkit/cars_meta.mat'

        train_loader = CarsDataset.load_data(mat_anno, data_dir, car_names)

        # create instance of net
        model = DenseNet161()

        # set two criterion loss
        criterion_label = nn.CrossEntropyLoss()
        criterion_bbox = nn.MSELoss()

        # set stochastic gradient descent?
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        # optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate, weight_decay=weight_decay)

        num_epochs = 1
        model.train(True)

        logger.info(' Start training ...')
        for batch, (images, labels, bbox) in enumerate(train_loader):
            logger.info(images)
            logger.info(labels)
            logger.info(bbox)

            break

