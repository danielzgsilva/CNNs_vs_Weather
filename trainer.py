import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torchvision.models import inception_v3, resnet34, vgg16_bn

from utils import *

import time


class ClassificationTrainer:
    def __init__(self, args):
        self.opt = args
        self.data_path = self.opt.data_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = self.opt.model_path
        self.model_name = self.opt.model_name
        self.model_type = self.opt.model_type

        # Training parameters
        self.input_size = (self.opt.height, self.opt.width)
        self.batch_size = int(self.opt.batch_size)
        self.num_workers = int(self.opt.num_workers)
        self.epochs = int(self.opt.num_epochs)
        self.lr = float(self.opt.learning_rate)
        self.step = int(self.opt.scheduler_step_size)

        # Create model and place on GPU
        assert self.model_type in ['inception', 'resnet', 'VGG']

        if self.model_type == 'inception':
            self.model = inception_v3(pretrained=True, aux_logits=False)
            self.model.fc = self.fc = nn.Linear(2048, len(important_classes))
        elif self.model_type == 'VGG':
            self.model = vgg16_bn(pretrained=True, num_classes=len(important_classes))
        else:
            self.model = resnet34(pretrained=True, num_classes=len(important_classes))

        self.model.to(self.device)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        # self.scheduler = StepLR(self.optimizer, step_size=self.step, gamma=0.9)

        print('Training options:\n'
              '\tInput size: {}\n\tBatch size: {}\n\tEpochs: {}\n\t'
              'Learning rate: {}\n\tStep Size: {}\n\tLoss: {}\n\tOptimizer: {}\n'. \
              format(self.input_size, self.batch_size, self.epochs, self.lr, self.step, self.criterion, self.optimizer))

        # Data transformations to be used during loading of images
        self.data_transforms = {'train': transforms.Compose([transforms.Resize(self.input_size),
                                                             transforms.ToTensor()]),
                                'val': transforms.Compose([transforms.Resize(self.input_size),
                                                           transforms.ToTensor()]),
                                'test': transforms.Compose([transforms.Resize(self.input_size),
                                                            transforms.ToTensor()])}

        # Creating PyTorch datasets
        self.datasets = dict()
        train_dataset = Cityscapes(self.data_path,
                                   split='train',
                                   mode='fine',
                                   target_type=["polygon"],
                                   transform=self.data_transforms['train'],
                                   target_transform=get_image_label)

        trainextra_dataset = Cityscapes(self.data_path,
                                        split='train_extra',
                                        mode='coarse',
                                        target_type=["polygon"],
                                        transform=self.data_transforms['train'],
                                        target_transform=get_image_label)

       # self.datasets['train'] = ConcatDataset([train_dataset, trainextra_dataset])
        self.datasets['train'] = train_dataset

        self.datasets['val'] = Cityscapes(self.data_path,
                                          split='val',
                                          mode='coarse',
                                          target_type=['polygon'],
                                          transform=self.data_transforms['val'],
                                          target_transform=get_image_label)

        self.datasets['test'] = Cityscapes(self.data_path,
                                           split='test',
                                           mode='fine',
                                           target_type=["instance", "semantic", "polygon", "color"],
                                           transform=self.data_transforms['test'],
                                           target_transform=get_image_label)

        self.dataset_lens = [self.datasets[i].__len__() for i in ['train', 'val', 'test']]

        print('Training on:\n'
              '\tTrain files: {}\n\tValidation files: {}\n\tTest files: {}\n' \
              .format(*self.dataset_lens))

        # Creating PyTorch dataloaders
        self.dataloaders = {i: DataLoader(self.datasets[i], batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers) for i in ['train', 'val']}

        self.test_loader = DataLoader(dataset=self.datasets['test'], batch_size=1, shuffle=True)

    def run_epoch(self, phase):
        running_loss = 0.0
        running_corrects = 0

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        # Looping through batches
        for i, (images, labels) in enumerate(self.dataloaders[phase]):

            # Ensure we're doing this calculation on our GPU if possible
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero parameter gradients
            self.optimizer.zero_grad()

            # Calculate gradients only if we're in the training phase
            with torch.set_grad_enabled(phase == 'train'):

                # This calls the forward() function on a batch of inputs
                outputs = self.model(images)

                # Calculate the loss of the batch
                loss = self.criterion(outputs, labels)

                # Gets the predictions of the inputs (highest value in the array)
                _, preds = torch.max(outputs, 1)

                # Adjust weights through backprop if we're in training phase
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            # Document statistics for the batch
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        # Calculate epoch statistics
        loss = running_loss / self.datasets[phase].__len__()
        acc = running_corrects / self.datasets[phase].__len__()

        return loss, acc

    def train(self):
        start = time.time()

        best_model_wts = self.model.state_dict()
        best_acc = 0.0

        print('| Epoch\t | Train Loss\t| Train Acc\t| Valid Loss\t| Valid Seq Acc\t| Valid Dig ' 'Acc\t| Epoch Time |')
        print('-' * 102)

        # Iterate through epochs
        for epoch in range(self.epochs):

            epoch_start = time.time()

            # Training phase
            train_loss, train_acc = self.run_epoch('train')

            # Validation phase
            val_loss, val_acc = self.run_epoch('val')

            epoch_time = time.time() - epoch_start

            # Print statistics after the validation phase
            print("| {}\t | {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.0f}m {:.0f}s     |"
                  .format(epoch + 1, train_loss, train_acc, val_loss, val_acc,
                          epoch_time // 60, epoch_time % 60))

            # Copy and save the model's weights if it has the best accuracy thus far
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = self.model.state_dict()

        total_time = time.time() - start

        print('-' * 118)
        print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
        print('Best validation accuracy: {:.4f}'.format(best_acc))

        # load best model weights and save them
        self.model.load_state_dict(best_model_wts)

        save_model(self.model_path, self.model_name, self.model, self.epochs, self.optimizer, self.criterion)

        return
