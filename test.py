import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import os
from options import TestingOptions
from utils import load_model, get_image_label
from datasets.cityscapes import Cityscapes

options = TestingOptions()
opts = options.parse()


class Tester:
    def __init__(self, args):
        self.opt = args
        self.data_path = self.opt.data_path

        self.model_path = self.opt.model_path
        self.model_name = self.opt.model_name

        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_workers
        self.input_size = (self.opt.height, self.opt.width)

        # Load model for testing
        model, optimizer, criterion, epoch, device = load_model(os.path.join(self.model_path, self.model_name))
        model.eval()
        self.model = model
        self.device = device
        self.criterion = criterion

        self.transforms = transforms.Compose([transforms.Resize(self.input_size),
                                              transforms.ToTensor()])

        # List of perturbations we'll test the model on
        self.perturbations = ['none', 'fog']

        # Dataset and dataloader dictionaries indexed by the type of perturbation it applies to images
        self.datasets = {i: Cityscapes(self.data_path,
                                       split='test',
                                       mode='fine',
                                       target_type=['polygon'],
                                       transform=self.transforms,
                                       target_transform=get_image_label,
                                       perturbation=i)
                         for i in self.perturbations}

        self.num_testing_files = self.datasets['none'].__len__()
        print('Testing on {} test files'.format(self.num_testing_files))

        # Creating PyTorch dataloaders
        self.dataloaders = {i: DataLoader(self.datasets[i], batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers) for i in self.perturbations}

    def test(self):
        print('| Perturbation\t| Loss\t| Accuracy\t|')
        print('-' * 42)

        # Test model on each type of perturbation
        for perturb in self.perturbations:
            running_loss = 0.0
            running_corrects = 0

            # Looping through batches
            for i, (images, labels) in enumerate(self.dataloaders[perturb]):

                # Ensure we're doing this calculation on our GPU if possible
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    # This calls the forward() function on a batch of inputs
                    outputs = self.model(images)

                    # Calculate the loss of the batch
                    loss = self.criterion(outputs, labels)

                    # Gets the predictions of the inputs (highest value in the array)
                    _, preds = torch.max(outputs, 1)

                # Document statistics for the batch
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            # Document statistics
            acc = running_corrects / self.num_testing_files
            loss = running_loss / self.num_testing_files

            # Print results
            print("| {}\t | {:.4f}\t| {:.4f}\t|".format(perturb, loss, acc,))

if __name__ == "__main__":
    trainer = Tester(opts)

    trainer.test()
