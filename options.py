import os
import argparse

file_dir = os.path.dirname(__file__)


class CityscapesOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Cityscape training options")

        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, 'cityscapes'))
        self.parser.add_argument("--model_path",
                                 type=str,
                                 help="directory to save model weights in",
                                 default=os.path.join(file_dir, "models"))

        self.parser.add_argument("--model_type",
                                 type=str,
                                 help="enter the type of CNN architecture you'd like to train",
                                 choices=['inception', 'resnet', 'VGG']
                                 )
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="name of the model")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=512)

        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=32)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=0.01)
        self.parser.add_argument("--num_epochs",
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=3)
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        self.parser.add_argument("--model_to_load",
                                 nargs="+",
                                 type=str,
                                 help="model to load")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
