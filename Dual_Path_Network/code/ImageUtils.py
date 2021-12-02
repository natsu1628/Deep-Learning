import numpy as np
from PIL import Image
import torchvision.transforms as transforms

"""This script implements the functions for data augmentation
and preprocessing.
"""


def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: A tensor of shape [3, 32, 32].
    """
    # convert the image from array of [3072,] to a 3-D image of [3, 32, 32] --> [depth, height, width]
    depth_major = record.reshape((3, 32, 32))
    # convert image into [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])
    # preprocess the training image (or testing image with normalization)
    image = preprocess_image(image, training)

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: A tensor of shape [3, 32, 32]. The processed image.
    """
    transform_image = None
    if training:
        transform_image = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
    else:
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
    image = Image.fromarray(image)
    image = transform_image(image)

    return image


def preprocess_test(image):
    """
    Preprocess a single private test data image
    Args:
        image: An array of shape [32, 32, 3]

    Returns:
        image: A processed image of tensor format with shape [3, 32, 32]
    """
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    image = Image.fromarray(image)
    image = transform_image(image)
    return image
