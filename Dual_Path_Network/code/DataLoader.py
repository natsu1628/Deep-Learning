import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""


def get_data(data_dir):
    """Get the CIFAR-10 data for training and testing

    Args:
        data_dir: A string. The directory where training/test data are stored

    Returns:
        x_batch: An numpy array of shape [train(test)_batch_size, 3072] containing images
            (dtype=np.float32)
        y_batch: An numpy array of shape [train(test)_batch_size, 3072] containing labels
            (dtype=np.float32)
    """
    x_list = list()
    y_list = list()
    for train_file in os.listdir(data_dir):
        with open(os.path.join(data_dir, train_file), 'rb') as tf:
            batch_data = pickle.load(tf, encoding='bytes')
            x_list.append(np.array(batch_data[b"data"]))
            y_list.append(np.array(batch_data[b"labels"]))
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return x, y


def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")

    # Load training data
    x_train, y_train = get_data(train_dir)
    print("Train data X and Y shape: x_train:", x_train.shape, ", y_train:", y_train.shape)

    # Load testing data
    x_test, y_test = get_data(test_dir)
    print("Test data X and Y shape: x_test:", x_test.shape, ", y_test:", y_test.shape)
    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    private_dir = os.path.join(data_dir, "Private")
    x_test = np.load(os.path.join(private_dir, 'private_test_images_v3.npy'))

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.9):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """

    train_split_index = int(x_train.shape[0] * train_ratio)
    x_train_new = x_train[:train_split_index]
    y_train_new = y_train[:train_split_index]

    x_valid = x_train[train_split_index:]
    y_valid = y_train[train_split_index:]

    return x_train_new, y_train_new, x_valid, y_valid
