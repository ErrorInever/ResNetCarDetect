import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
import time
import torch.optim as optim
from predata import CarsDataset
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")


class DenseNet161(nn.Module):

    def __init__(self, num_classes=196, drop_rate=0):
        super(DenseNet161, self).__init__()

        original_model = models.densenet161(pretrained=True, drop_rate=drop_rate)
        # get all layers from original model without last layer
        # and create our custom layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])

        # x4?
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

cars_data = CarsDataset(mat_anno, data_dir, car_names,
                        transform=transforms.Compose([
                            transforms.Scale(250),
                            transforms.RandomSizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4706145, 0.46000465, 0.45479808),
                                                 (0.26668432, 0.26578658, 0.2706199))
                        ])
                        )


train_loader = DataLoader(cars_data, batch_size=1, shuffle=True)

print(len(cars_data))

# start_time = time.time()
model = DenseNet161()
# x = torch.randn(1, 3, 512, 640)
# output_labels, output_bbox = model(x)
# print('finish time: {0}m'.format((time.time() - start_time)/60))

# two criterion
criterion_label = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate, weight_decay=weight_decay)

num_epochs = 1
print("Start training")
model.train(True)


for batch, (images, labels, bbox) in enumerate(train_loader):
    print(images)
    print(labels)
    print(bbox)
    break
