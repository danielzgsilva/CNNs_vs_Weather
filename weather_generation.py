import numpy as np
import cv2 as cv
import os
from PIL import Image

import torch


def add_fog(im, D, tFactor, atmLight):
    """
    Adds synthetic fog to an image
    im : image to add synthetic fog, should be a torch tensor
    D : image's corresponding depth map
    tFactor : fog thickness factor
    atmLight : atmospheric light
    """
    tFactor *= 100

    foggy = im.clone()

    # Add fog
    n, m = foggy.shape[1:]
    for i in range(n):
        for j in range(m):
            # Compute transmission
            t = np.exp(-tFactor / D[i, j].item())
            print(t)
            # Set intensity of fog
            print(foggy[:, i, j])
            foggy[:, i, j] = t * foggy[:, i, j] + ((1 - t) * atmLight)

    # Return foggy image
    return foggy