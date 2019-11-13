"""
Wrapper over Neural Net
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models import Net
from utils import NUM_CHANNELS, TRANSFORM


class NNet():
    """
    Wrapper to manage neural net.
    """

    def __init__(self, args):
        self.args = args
        self.num_channels = NUM_CHANNELS
        self.net = Net(self.num_channels)
        if self.args.cuda:
            self.net = self.net.cuda()
        self.load_dataset_from_folder()
        self.writer = SummaryWriter()
        self.unique_tok = str(time.time())

    def load_dataset_from_folder(self):
        """
        Load complete dataset
        """
        all_data_path = self.args.all_data_path
        validation_split_size = self.args.validation_split_size
        batch_size = self.args.batch_size
        num_workers = self.args.num_workers
        shuffle = self.args.shuffle

        all_data = ImageFolder(
            root=all_data_path,
            transform=TRANSFORM
        )

        classes = all_data.classes
        self.classes = classes

        validation_size = int(validation_split_size * len(all_data))
        test_size = int(validation_split_size * len(all_data))
        train_size = len(all_data) - 2*validation_size
        train_dataset, val_dataset, test_dataset = random_split(
            all_data, [train_size, validation_size, test_size])

        training_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )

        validation_dataset_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )

        test_dataset_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )

        self.train_loader = training_data_loader
        self.val_loader = validation_dataset_loader
        self.test_loader = test_dataset_loader

    def train(self):
        """
        Train Neural Net
        """
        optimizer = optim.RMSprop(self.net.parameters(), lr=self.args.lr,
                                  momentum=self.args.momentum, weight_decay=self.args.l2_regularization)
        criterion = nn.CrossEntropyLoss()

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)

        self.net.train()

        for epoch in range(self.args.epoch):
            start_time = time.time()

            running_loss_t = 0.0
            num_batches = 0

            for data in tqdm(self.train_loader):
                inputs, labels = data

                if self.args.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = self.net(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss_t += loss.item()
                num_batches += 1

            end_time = time.time()

            scheduler.step()

            self.save(epoch+1)
            self.writer.add_scalar(
                'Loss/train', running_loss_t/num_batches, epoch+1)

            self.net.eval()
            loss_v = self.get_validation_loss(criterion)
            loss_v = 0
            self.net.train()
            self.writer.add_scalar('Loss/val', loss_v, epoch+1)

            print("Epoch {} Time {:.2f}s Training-Loss {:.3f} Validation-Loss {:.3f}".format(
                epoch+1, end_time-start_time, running_loss_t/num_batches, loss_v))

    def get_validation_loss(self, criterion):
        """
        Check validation loss
        """
        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data in tqdm(self.val_loader):
                images, labels = data

                if self.args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = self.net(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                num_batches += 1

        return running_loss/num_batches

    def get_test_accuracy(self):
        """
        Check overall accuracy of model
        """
        y_true = []
        y_pred = []
        class_correct = list(0. for i in range(4))
        class_total = list(0. for i in range(4))

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                images, labels = data
                labels_cp = labels.clone()
                if self.args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu()
                for i, pred in enumerate(predicted):
                    y_pred.append(pred)
                    y_true.append(labels_cp[i])
                c = (predicted == labels_cp).squeeze()

                for i in range(min(self.args.batch_size, len(labels_cp))):
                    label = labels_cp[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        print("Test F1: ", f1_score(y_true, y_pred, average='weighted'))

    def save(self, epochs, folder_path="../models/"):
        """
        Save Model
        """
        dict_save = {
            'params': self.net.state_dict(),
            'classes': self.classes
        }
        name = folder_path + self.unique_tok + '_' + str(epochs)+'.model'
        torch.save(dict_save, name)
        print('Model saved at {}'.format(name))
        return name

    def load(self, path):
        """
        Load a saved model
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict_load = torch.load(path, map_location=torch.device(device))
        self.net.load_state_dict(dict_load['params'])
        return dict_load['classes']

    def predict(self, inp):
        """
        Predict using net
        """

        if self.args.cuda:
            inp = inp.cuda()

        self.net.eval()
        with torch.no_grad():
            vals = self.net(inp)
            _, predicted = torch.max(vals, 1)
            predicted = predicted.cpu()
            result_class = self.classes[predicted]

        return result_class
