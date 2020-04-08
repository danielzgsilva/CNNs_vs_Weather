import numpy as np
import cv2 as cv
import os
from PIL import Image
from scipy import special

import torch
from torchvision import transforms

def add_fog(im, D, tFactor, atmLight):
    """
    Adds synthetic fog to an image
    im : image to add synthetic fog, should be a torch tensor
    D : image's corresponding depth map
    tFactor : fog thickness factor
    atmLight : atmospheric light
    """
    tFactor *= 100

    im = im.numpy()
    foggy = np.copy(im)

    # Add fog
    c, h, w = foggy.shape
    for i in range(h):
        for j in range(w):
            # Compute transmission
            t = special.expit(-tFactor / D[i, j])

            # Set intensity of fog
            foggy[:, i, j] = t * foggy[:, i, j] + ((1 - t) * atmLight)
    
    return torch.from_numpy(foggy)