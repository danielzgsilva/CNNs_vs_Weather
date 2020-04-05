from options import CityscapesOptions
from trainer import ClassificationTrainer

options = CityscapesOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = ClassificationTrainer(opts)

    trainer.train()