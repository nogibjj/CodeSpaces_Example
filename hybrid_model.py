import os
import urllib
import zipfile
import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def prep_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    # Convert training and test data to TensorDatasets
    trainset = TensorDataset(
        torch.from_numpy(np.array(X_train)).long(),
        torch.from_numpy(np.array(y_train)).float(),
    )
    valset = TensorDataset(
        torch.from_numpy(np.array(X_val)).long(),
        torch.from_numpy(np.array(y_val)).float(),
    )

    # Create Dataloaders for our training and test data to allow us to iterate over minibatches
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )

    return trainloader, valloader


class NNHybridFiltering(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        n_sentiments,
        embdim_users,
        embdim_items,
        embdim_u_sentiments,
        embdim_i_sentiments,
        n_activations,
        rating_range,
    ):
        super().__init__()
        self.user_embeddings = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embdim_users
        )
        self.item_embeddings = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embdim_items
        )
        self.u_sentiment_embeddings = nn.Embedding(
            num_embeddings=n_sentiments, embedding_dim=embdim_u_sentiments
        )
        self.i_sentiment_embeddings = nn.Embedding(
            num_embeddings=n_sentiments, embedding_dim=embdim_i_sentiments
        )
        self.fc1 = nn.Linear(
            embdim_users + embdim_items + embdim_u_sentiments + embdim_i_sentiments,
            n_activations,
        )
        self.fc2 = nn.Linear(n_activations, 1)
        self.rating_range = rating_range

    def forward(self, X):
        # Get embeddings for minibatch
        embedded_users = self.user_embeddings(X[:, 0])
        embedded_items = self.item_embeddings(X[:, 1])
        embedded_u_sentiments = self.u_sentiment_embeddings(X[:, 2])
        embedded_i_sentiments = self.i_sentiment_embeddings(X[:, 2])
        # Concatenate user, item and genre embeddings
        embeddings = torch.cat(
            [
                embedded_users,
                embedded_items,
                embedded_u_sentiments,
                embedded_i_sentiments,
            ],
            dim=1,
        )
        # Pass embeddings through network
        preds = self.fc1(embeddings)
        preds = F.relu(preds)
        preds = self.fc2(preds)
        # Scale predicted ratings to target-range [low,high]
        preds = (
            torch.sigmoid(preds) * (self.rating_range[1] - self.rating_range[0])
            + self.rating_range[0]
        )
        return preds


def train_model(
    model, criterion, optimizer, dataloaders, device, num_epochs=5, scheduler=None
):
    model = model.to(device)  # Send model to GPU if available
    since = time.time()

    costpaths = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Get the inputs and labels, and send to GPU if available
            for (inputs, labels) in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the weight gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model.forward(inputs).view(-1)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == "train":
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # Convert loss into a scalar and add it to running_loss
                running_loss += np.sqrt(loss.item()) * labels.size(0)

            # Step along learning rate scheduler when in train
            if (phase == "train") and (scheduler is not None):
                scheduler.step()

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            costpaths[phase].append(epoch_loss)
            print("{} loss: {:.4f}".format(phase, epoch_loss))

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return costpaths


if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/tfidf_train.csv")
    test_df = pd.read_csv("data/processed/tfidf_test.csv")
    df = train_df.append(test_df)

    encoder = LabelEncoder()
    encoder.fit(df["ProductId"])
    df["ProductId"] = encoder.transform(df["ProductId"])

    encoder = LabelEncoder()
    encoder.fit(df["UserId"])
    df["UserId"] = encoder.transform(df["UserId"])

    df["sentiment"] = df["sentiment"].map(lambda x: int(x) + 1)

    X = df.loc[:, ["UserId", "ProductId", "sentiment"]]
    y = df.loc[:, "Score"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, random_state=0, test_size=0.2
    )

    batchsize = 64
    trainloader, valloader = prep_dataloaders(X_train, y_train, X_val, y_val, batchsize)

    dataloaders = {"train": trainloader, "val": valloader}
    n_users = X.loc[:, "UserId"].max() + 1
    n_items = X.loc[:, "ProductId"].max() + 1
    n_sentiments = X.loc[:, "sentiment"].max() + 1
    model = NNHybridFiltering(
        n_users,
        n_items,
        n_sentiments,
        embdim_users=50,
        embdim_items=50,
        embdim_u_sentiments=25,
        embdim_i_sentiments=25,
        n_activations=100,
        rating_range=[0.0, 5.0],
    )
    criterion = nn.MSELoss()
    lr = 0.001
    n_epochs = 10
    wd = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cost_paths = train_model(
        model, criterion, optimizer, dataloaders, device, n_epochs, scheduler=None
    )
    torch.save(model, "models/fullmodel.pt")
