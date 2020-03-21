import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 128
nb_classes = 10
nb_epoch = 25
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

# Use cuda if available
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # TODO initialize model layers here
        self.conv1 = nn.Conv2d(1, 128, (5, 5))
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(128, 64, (3, 3))
        self.pool2_1 = nn.MaxPool2d((2, 2))
        self.conv2_2 = nn.Conv2d(128, 64, (3, 3))
        self.pool2_2 = nn.MaxPool2d((2, 2))
        self.conv3_1 = nn.Conv2d(32, 16, (3, 3))
        self.pool3_1 = nn.MaxPool2d((2, 2))
        self.conv3_2 = nn.Conv2d(32, 16, (3, 3))
        self.pool3_2 = nn.MaxPool2d((2, 2))


        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.linear1_1 = nn.Linear(2560, 256)
        self.linear1_2 = nn.Linear(2560,256)
        self.drop_out1 = nn.Dropout(0.3)
        self.drop_out2 = nn.Dropout(0.3)
        self.linear2_1 = nn.Linear(256, 128)
        self.linear2_2 = nn.Linear(256, 128)
        self.out1 = nn.Linear(128, 10)
        self.out2 = nn.Linear(128, 10)
    def forward(self, x):
        x.cuda(device)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.pool1(x)

        # Apply different net branch for each digit
        x_1 = self.conv2_1(x)
        x_1 = F.leaky_relu(x_1)
        x_1 = self.pool2_1(x_1)
        x_1 = self.flatten1(x_1)
        x_1 = self.linear1_1(x_1)
        x_1 = F.leaky_relu(x_1)
        x_1 = self.drop_out1(x_1)
        x_1 = self.linear2_1(x_1)
        x_1 = F.leaky_relu(x_1)

        x_2 = self.conv2_2(x)
        x_2 = F.leaky_relu(x_2)
        x_2 = self.pool2_2(x_2)
        x_2 = self.flatten2(x_2)
        x_2 = self.linear1_2(x_2)
        x_2 = F.leaky_relu(x_2)
        x_2 = self.drop_out2(x_2)
        x_2 = self.linear2_2(x_2)
        x_2 = F.leaky_relu(x_2)

        # Make prediction
        out_first_digit, out_second_digit = F.softmax(self.out1(x_1)), F.softmax(self.out2(x_2))

        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension).cuda(torch.device('cuda:0')) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model,n_epochs=nb_epoch)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
