import torch
import torch.nn as nn
import os, time
import numpy as np
from tqdm import tqdm
from Network import DualPathNetwork
from ImageUtils import parse_record, preprocess_test

"""This script defines the training, validation and testing process.
"""


class CifarDL(nn.Module):
    def __init__(self, config):
        super(CifarDL, self).__init__()
        self.config = config
        self.network = DualPathNetwork(config)
        self.network = self.network.cuda()
        # define cross entropy loss and optimizer
        self.learning_rate = 0.01
        self.crossEntropyloss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=0.9,
                                         weight_decay=self.config.weight_decay)
        # adjust learning rate using scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def model_setup(self):
        pass

    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size
        loss = 0

        print('### Training... ###')
        for epoch in range(1, max_epoch + 1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            for i in range(num_batches):
                # Construct the current batch.
                curr_x_batch = [parse_record(x, True) for x in
                                curr_x_train[i * self.config.batch_size: (i + 1) * self.config.batch_size]]
                curr_y_batch = curr_y_train[i * self.config.batch_size: (i + 1) * self.config.batch_size]
                curr_x_batch_tensor = torch.stack((curr_x_batch)).float().cuda()
                curr_y_batch_tensor = torch.tensor((curr_y_batch)).float().cuda()
                # call the model to get the output
                self.model = self.network.cuda()
                outputs = self.model(curr_x_batch_tensor)

                self.optimizer.zero_grad()
                loss = self.crossEntropyloss(outputs, curr_y_batch_tensor.long())
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            self.scheduler.step()

            if epoch % self.config.save_interval == 0:
                self.save(epoch)

    def evaluate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.save_dir, 'model-dpn-%d.ckpt' % checkpoint_num)
            self.load(checkpointfile)

            preds = []
            for i in tqdm(range(x.shape[0])):
                curr_x = x[i]
                device = 'cuda'
                curr_x = parse_record(curr_x, False).float().to(device)
                predict_output = self.network(curr_x.view(1, 3, 32, 32))
                predict = int(torch.max(predict_output.data, 1)[1])
                preds.append(predict)

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds == y) / y.shape[0]))

    def predict_prob(self, x):
        self.network.eval()
        print('### Private Data Test ###')
        model_name = self.config.checkpoint_name + '.ckpt'
        best_model_path = os.path.join(self.config.private_model, model_name)
        self.load(best_model_path)

        predictions = []
        for i in tqdm(range(x.shape[0])):
            curr_x = x[i].reshape((32, 32, 3))
            device = 'cuda'
            inputs = preprocess_test(curr_x).float().to(device)
            inputs = inputs.view(1, 3, 32, 32)
            output = self.network(inputs)
            predictions.append(output.cpu().detach().numpy())

        # converting the result of predictions into probabilities
        predictions = np.array(predictions)
        # converting the shape from (2000, 1, 10) to (2000, 10)
        predictions = predictions.reshape((predictions.shape[0], predictions.shape[1]*predictions.shape[2]))
        predictions_exp = np.exp(predictions)
        pred_exp_sum = predictions_exp.sum(axis=1)
        predictions_proba = (predictions_exp.T/pred_exp_sum).T

        return np.array(predictions_proba)

    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.save_dir, 'model-dpn-%d.ckpt' % epoch)
        os.makedirs(self.config.save_dir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))
