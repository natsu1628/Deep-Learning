import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        ### YOUR CODE HERE
        self.network = self.network.cuda()
        # define cross entropy loss and optimizer
        self.learning_rate = 0.1
        self.crossEntropyloss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.config.weight_decay)
        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        # print("Max epoch: ", max_epoch)
        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            # print("Epoch stated: ", epoch)
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.learning_rate = 0.1
            if epoch % 50 == 0:
                self.learning_rate = self.learning_rate / 10
            loss = 0
            ### YOUR CODE HERE
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                curr_x_batch = [parse_record(x, True) for x in curr_x_train[i * self.config.batch_size: (i + 1) * self.config.batch_size]]
                curr_y_batch = curr_y_train[i * self.config.batch_size: (i + 1) * self.config.batch_size]
                curr_x_batch_tensor = torch.tensor(curr_x_batch).float().cuda()
                curr_y_batch_tensor = torch.tensor(curr_y_batch).float().cuda()
                # call the model to get the output
                # model = self.network().to(device)
                # print("Batch size: ", curr_x_batch_tensor.size())
                self.model = self.network.cuda()
                # print("x batch tensor size: ", curr_x_batch_tensor.size())
                outputs = self.model(curr_x_batch_tensor)
                # print("Outputs returned for batch ", i)
                loss = self.crossEntropyloss(outputs, curr_y_batch_tensor.long())
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            if self.config.resnet_version == 1:
                res = "std-res"
            else:
                res = "bottleneck-res"
            checkpointfile = os.path.join(self.config.modeldir, 'model-%s-%d.ckpt'%(res, checkpoint_num))
            self.load(checkpointfile)

            preds = []
            # print("Shape: ", x.shape[0])
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                curr_x = x[i]
                curr_x = torch.tensor(parse_record(curr_x, False)).float().cuda()
                # self.network = self.network.cpu()
                predict_output = self.network(curr_x.view(1, 3, 32, 32))
                # print("Predict output: ", (torch.max(predict_output.data, 1)))
                predict = int(torch.max(predict_output.data, 1)[1])
                # print("Predict: ", predict)
                preds.append(predict)
                ### END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
    
    def save(self, epoch):
        if self.config.resnet_version == 1:
            res = "std-res"
        else:
            res = "bottleneck-res"
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%s-%d.ckpt'%(res, epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))
