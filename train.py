from options import TrainingOptions
from trainer import ClassificationTrainer

options = TrainingOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = ClassificationTrainer(opts)

    trainer.train()