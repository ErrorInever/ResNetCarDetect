import logging
import warnings
import torch
import time
import tqdm
import torch.nn.functional as F
from torch import nn
from torchvision import models
import torch.optim as optim
from predata import CarsDataset
from torch.autograd import Variable



warnings.filterwarnings("ignore")
module_logger = logging.getLogger("main.model")


class DenseNet161(nn.Module):

    def __init__(self, num_classes=196, drop_rate=0):
        super(DenseNet161, self).__init__()

        # create base model
        original_model = models.densenet161(pretrained=False, drop_rate=drop_rate)
        # get all layers from original model without last layer
        # and create our custom layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
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
    def train_model(params):
        """
        training of net
        :param params: args from command line
        """
        logger = logging.getLogger('main.model.Densenet161.train')
        # get params of cmd
        data_dir = params["data_dir"]
        cuda = params["cuda"]
        num_epochs = params["epochs"]
        lr = params["lr"]
        batch_size = params["bs"]
        workers = params["workers"]
        losswise_key = params["key"]

        # create data loader
        train_loader = CarsDataset.load_data(data_dir)

        if cuda:
            # create instance of net
            model = DenseNet161().cuda()

            # create two loss function
            criterion_label = nn.CrossEntropyLoss().cuda()
            criterion_bbox = nn.MSELoss().cuda()
        else:
            model = DenseNet161()
            criterion_label = nn.CrossEntropyLoss()
            criterion_bbox = nn.MSELoss()

        # create optimizer with stochastic gradient descent / adam
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        # optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate, weight_decay=weight_decay)

        logger.info(' Start training ...')

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # set model on training mode
            model.train(True)

            running_loss = 0.0
            c = 0
            correct_label = 0
            correct_bbox = 0

            total_label = 0
            total_bbox = 0

            for batch in tqdm.tqdm(train_loader):
                c += 1
                images, labels, bbox = batch
                labels = labels.type(torch.LongTensor)

                if cuda:
                    images, labels, bbox = Variable(images).cuda(), Variable(labels).cuda(), Variable(bbox).cuda().float()

                else:
                    images, labels, bbox = Variable(images), Variable(labels), Variable(bbox).float()

                # zero gradient
                optimizer.zero_grad()
                # input data through network
                outputs_label, outputs_bbox = model(images)

                # return index of max value of tensor
                _, predicted_label = torch.max(outputs_label.data, 1)
                _, predicted_bbox = torch.max(outputs_bbox.data, 1)

                # sends outputs of model to loss functions
                loss_label = criterion_label(outputs_label, labels)
                loss_bbox = criterion_bbox(outputs_bbox, bbox)
                loss = loss_label + loss_bbox

                # backward
                loss.backward()

                # do step to gradient
                optimizer.step()

                running_loss += loss.item()

                total_label += labels.size(0)
                total_bbox += bbox.size(0)

                print(predicted_label)
                print(labels.data)
                correct_label += (predicted_label == labels.data).sum()
                correct_bbox += (outputs_bbox == bbox.data).sum()

                if c == 10: break

            epoch_loss = running_loss / c

            train_acc_label = 100 * correct_label / total_label
            train_acc_bbox = 100 * correct_bbox / total_bbox

            print('Train epoch: {} || Loss: {:.4f} || Acc label: {:.2f} %%'.format(
                epoch, epoch_loss, train_acc_label))
            print('Acc bbox: {:.2f} %%'.format(train_acc_bbox))
