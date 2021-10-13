import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
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
    ### YOUR CODE HERE
    # data directory for train and test data
    train_dir = data_dir + "Train/"
    test_dir = data_dir + "Test/"

    # load training data
    x_train_list = list()
    y_train_list = list()
    for train_file in os.listdir(train_dir):
        with open(train_dir + train_file, 'rb') as tf:
            batch_data = pickle.load(tf, encoding='bytes')
            x_train_list.append(np.array(batch_data[b"data"]))
            y_train_list.append(np.array(batch_data[b"labels"]))
    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    print("Train data X and Y shape: x_train:", x_train.shape, ", y_train:", y_train.shape)

    # load test data
    x_test_list = list()
    y_test_list = list()
    for test_file in os.listdir(test_dir):
        with open(test_dir + test_file, 'rb') as tf:
            batch_data = pickle.load(tf, encoding='bytes')
            x_test_list.append(np.array(batch_data[b"data"]))
            y_test_list.append(np.array(batch_data[b"labels"]))

    x_test = np.concatenate(x_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    print("Test data X and Y shape: x_test:", x_test.shape, ", y_test:", y_test.shape)
    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid
