import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""


def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image


def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        h_old, w_old, d = image.shape
        # create an image with all pixels as 0 and then add the original image from (4:36, 4:36) section (array
        # numbering starts from 0)
        new_padded_image = np.full((h_old + 8, w_old + 8, d), fill_value=0)
        # add the original image into the padded image of pixel 0
        new_padded_image[4:4 + h_old, 4:4 + w_old] = image
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        rand_left_point = np.random.randint(9, size=2)
        crop_image = new_padded_image[rand_left_point[1]:rand_left_point[1]+32, rand_left_point[0]:rand_left_point[0]+32]
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        check_flip = np.random.randint(2)
        if check_flip == 1:
            crop_image = np.fliplr(crop_image)
        image = crop_image
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    np.seterr(divide='ignore', invalid='ignore')
    # Subtract off the mean and divide by the standard deviation of the pixels.
    # calculating mean and standard deviation of the entire image that includes all channels column wise
    # AlexNet
    # image = (image - np.mean(image, axis=(0, 1, 2), keepdims=True)) / (np.std(image, axis=(0, 1, 2), keepdims=True))
    # VGGNet - per channel mean
    image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / (np.std(image, axis=(0, 1), keepdims=True))
    image = np.nan_to_num(image, nan=0.0)
    ### YOUR CODE HERE

    return image
