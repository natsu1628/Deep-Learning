# import torch
import os, argparse
import numpy as np
from Model import CifarDL
from DataLoader import load_data, train_valid_split, load_testing_images
# from Configure import model_configs, training_configs, testing_configs, prediction_configs


def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="train, test or predict", default="train")
    parser.add_argument("--data_dir", type=str, help="path to the data",
                        default="../Data/")
    parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
    parser.add_argument("--save_dir", type=str, help="path to save the models",
                        default="../model_dir")
    parser.add_argument("--result_dir", type=str, help="path to save the private results",
                        default="../results")
    parser.add_argument("--weight_decay", type=float, help="weight decay to be used in Optimizer", default=5e-4)
    parser.add_argument("--save_interval", type=int, default=5,
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--private_model", type=str, help="path of best model for private data test",
                        default="../saved_models")
    parser.add_argument("--checkpoint_list", type=str, help="checkpoint list to be used for testing", default="50,75,"
                                                                                                              "100")
    parser.add_argument("--checkpoint_name", type=str, help="name of the best model checkpoint",
                        default="model-dpn")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = configure()
    model = CifarDL(config).cuda()

    if config.mode == 'train':
        x_train, y_train, x_test, y_test = load_data(config.data_dir)
        x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train, 1)

        print("New Train data X and Y shape for final training: x_train:", x_train.shape, ", y_train:", y_train.shape)
        max_epoch = 200
        model.train(x_train, y_train, max_epoch)
        # checkpoint_num_list = [int(x.strip()) for x in config.checkpoint_list.split(",")]
        # model.evaluate(x_valid, y_valid, checkpoint_num_list)

    elif config.mode == 'test':
        # Testing on public testing dataset
        checkpoint_num_list = [int(x.strip()) for x in config.checkpoint_list.split(",")]
        _, _, x_test, y_test = load_data(config.data_dir)
        model.evaluate(x_test, y_test, checkpoint_num_list)

    elif config.mode == 'predict':
        # Predicting and storing results on private testing dataset
        x_test = load_testing_images(config.data_dir)
        # create result directory if does not exist
        os.makedirs(config.result_dir, exist_ok=True)
        predictions = model.predict_prob(x_test)
        # print("Predictions shape: ", predictions.shape)
        np.save(os.path.join(config.result_dir, 'predictions.npy'), predictions)
        print("Predictions saved")
