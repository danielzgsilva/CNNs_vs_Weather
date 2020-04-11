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
    im : image to add synthetic fog, should be a PIL image
    D : image's corresponding depth map
    tFactor : fog thickness factor
    atmLight : atmospheric light
    """

    foggy = np.array(im).astype(np.float32) / 255

    # Add fog
    h, w, c = foggy.shape
    for i in range(h):
        for j in range(w):
            # Compute transmission
            t = special.expit(-tFactor / D[j, i])

            # Set intensity of fog
            foggy[i, j, :] = t * foggy[i, j, :] + ((1 - t) * atmLight)

    foggy = (foggy * 255).astype(np.uint8)
    return Image.fromarray(foggy)


def generate_random_lines(imshape, slant, drop_length):
    drops = []
    for i in range(1500):
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)

        y = np.random.randint(0, imshape[0] - drop_length)

        drops.append((x, y))
    return drops


def add_rain(im, D, tFactor, atmLight):
    """
    Adds synthetic rain to an image
    im : image to add synthetic rain, should be a PIL image
    D : image's corresponding depth map
    tFactor : rain thickness factor
    atmLight : atmospheric light
    """

    rainy = np.array(im)
    imshape = rainy.shape

    slant_extreme = 10
    slant = np.random.randint(-slant_extreme, slant_extreme)
    drop_length = 20
    drop_width = 2
    drop_color = (200, 200, 200)

    rain_drops = generate_random_lines(imshape, slant, drop_length)

    for rain_drop in rain_drops:
        rainy = cv.line(rainy, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length),
                        drop_color, drop_width)

    # rainy views are blurry are slightly dark
    rainy = (cv.blur(rainy, (7, 7)) * atmLight).astype(np.uint8)

    return Image.fromarray(rainy)


def add_snow(im, D, tFactor, atmLight):
    """
    Adds synthetic snow to an image
    im : image to add synthetic snow, should be a PIL image
    D : image's corresponding depth map
    tFactor : snow thickness factor
    atmLight : atmospheric light

    """

    snowy = np.array(im)
    h, w, c = snowy.shape

    # increase this for more snow
    snow_point = np.random.uniform(0, 1)
    snow_point *= 255 / 2
    snow_point += 255 / 3

    # How bright to make the snow
    brightness_coefficient = 2.5

    # Conversion to HLS
    image_HLS = cv.cvtColor(snowy, cv.COLOR_RGB2HLS)
    image_HLS = np.array(image_HLS, dtype=np.float64)

    # scale pixel values up for channel 1(Lightness)
    image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] = image_HLS[:, :, 1][image_HLS[:, :,
                                                                             1] < snow_point] * brightness_coefficient

    # Sets all values above 255 to 255
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)

    # Conversion to RGB
    snowy = cv.cvtColor(image_HLS, cv.COLOR_HLS2RGB).astype(np.uint8)

    return Image.fromarray(snowy)


def add_occlusion(im):
    """
    Adds a synthetic occlusion to an image
    im : image to add occlusion to, should be a torch tensor
    """

    image = np.array(im)
    imshape = image.shape

    x = np.random.randint(imshape[1])
    y = np.random.randint(imshape[0])
    center_coordinates = (x, y)

    x_axis_len = np.random.randint(100, 1000)
    y_axis_len = np.random.randint(100, 1000)
    axesLength = (x_axis_len, y_axis_len)

    angle = np.random.randint(0, 180)

    color = (0, 0, 0)

    startAngle = 0
    endAngle = 360

    # Line thickness of -1 px
    thickness = -1

    # Draw occlusion
    occluded_image = cv.ellipse(image, center_coordinates, axesLength, angle,
                                startAngle, endAngle, color, thickness)

    return Image.fromarray(occluded_image)
