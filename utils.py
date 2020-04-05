import torch
import os


important_classes = ['unlabeled',
                     'ground',
                     'road',
                     'sidewalk',
                     'parking',
                     'rail track',
                     'building',
                     'fence',
                     'guard rail',
                     'bridge',
                     'tunnel',
                     'person',
                     'rider',
                     'car',
                     'truck',
                     'bus',
                     'caravan',
                     'trailer',
                     'train',
                     'motorcycle',
                     'bicycle']

class_to_num = {label: i for i, label in enumerate(important_classes)}


def get_image_label(poly):
    objects = poly['objects']

    image_label = 'unlabeled'
    max_count = 0
    label_counts = {}

    for obj in objects:
        label = obj['label']

        if label in important_classes:
            label_counts[label] = label_counts.get(label, 0) + 1

            if label_counts[label] > max_count:
                max_count = label_counts[label]
                image_label = label

    label_num = class_to_num[image_label]

    target = torch.zeros(len(important_classes), dtype=torch.LongTensor)
    target[label_num] = 1
    return target


def save_model(path, name, arch, model, epochs, optimizer, criterion):
    model_path = os.path.join(path, name) + '.tar'

    torch.save({
        'model': arch,
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer,
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion': criterion
    }, model_path)


def load_model(filepath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = checkpoint['criterion']
    epoch = checkpoint['epoch']
    model.to(device)

    return model, optimizer, criterion, epoch, device
