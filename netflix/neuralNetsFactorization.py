"""Matrix factorization using deep neural network"""

import numpy as np
import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")


NUM_EPOCH = 600
BATCH_SIZE = 64
LR = 0.1


class DMF(nn.Module):
    """
    A simple neural network that takes one-hot
    user and movie pair as input and outputs
    the ranking
    """

    def __init__(self, num_users, num_movies, hidden_size=3, num_latent=3):
        super(DMF, self).__init__()
        self.user_encoder = nn.Linear(num_users, num_latent)
        self.movie_encoder = nn.Linear(num_movies, num_latent)
        self.perceptron = nn.Linear(num_latent * 2, 1)

    def forward(self, u, v):
        user_latent = F.relu(self.user_encoder(u))
        movie_latent = F.relu(self.movie_encoder(v))

        feature = torch.cat([user_latent, movie_latent], dim=1)
        p1 = F.relu(self.perceptron(feature))
        return p1

def run_epoch(optimizer, model, ratings):
    """
    Train the  model on the given data set
    Args:
        data: np.ndarray: one-hot encoding

    Returns:

    """

    loss_list = []
    num_u, num_v = ratings.shape
    # Train model through one pass of data
    user_seq = np.random.permutation(num_u)
    for u in user_seq:
        true_ratings = torch.from_numpy(ratings[u, :].squeeze()).float()
        # Generate batch consist of one-hot encoding of num_movie duplicates of one user versus all movies
        u_data = generate_n_one_hot(u, num_u, num_v).float()
        v_data = torch.eye(num_v).float()

        predictions = model(u_data, v_data).squeeze()

        # Consider only observed data
        loss = F.mse_loss(predictions[true_ratings != 0], true_ratings[true_ratings != 0])
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_list


def train_run():
    num_users, num_movies = X_gold.shape
    model = DMF(num_users, num_movies)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    for _ in range(NUM_EPOCH):
        loss_l = run_epoch(optimizer, model, X)
        print(np.mean(loss_l))

    return model

def generate_n_one_hot(position, dim, n_duplicates):
    one_hot = np.zeros(dim, dtype='float32')
    one_hot[position] = 1
    one_hot_dup = np.tile(one_hot, (n_duplicates, 1))
    return torch.from_numpy(one_hot_dup)



if __name__ == "__main__":
    train_run()

