import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader, TensorDataset
import torch.optim as optim
from Loss import *
from sklearn.decomposition import PCA
import os
import pandas as pd
from sklearn import preprocessing
import numpy as np
from Model import *
from ArgReader import *


if __name__ == "__main__":
    # Torch setting
    torch.set_default_dtype(torch.float64)

    # Parameters
    data_path = "/Users/jacobvons/Downloads/Y4S1/COMP4540/federated_learning/all.csv"
    reader = ArgReader("./test_args.csv")
    reader.parse()
    for args in reader.args:
        epoch_num = int(args["e"])
        comm_rounds = int(args["rounds"])
        name = args["name"]
        explain_ratio = float(args["ratio"])
        lr = float(args["lr"])

        # Data Split
        data_df = pd.read_csv(data_path)
        features = data_df[data_df.columns[:-1]]
        features = preprocessing.normalize(features, axis=0)
        targets = np.array(data_df[data_df.columns[-1]])
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

        # PCA
        pca = PCA(n_components=5)
        while True:
            pca.fit(X_train)
            if sum(pca.explained_variance_ratio_) >= explain_ratio:
                pc_num = len(pca.components_)
                break
            else:
                pca = PCA(n_components=pca.n_components + 1)
        print("At least", pc_num, "PCs")

        # Dimension Reduction
        if not os.path.exists("./central"):
            os.mkdir("./central")
        if not os.path.exists(f"./central/{name}"):
            os.mkdir(f"./central/{name}")
        dir_path = f"./central/{name}"
        pcs = pca.components_
        reduced_X_train = torch.from_numpy(X_train @ pcs.T)
        reduced_X_test = torch.from_numpy(X_test @ pcs.T)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)
        train_dataset = TensorDataset(reduced_X_train, y_train)
        test_dataset = TensorDataset(reduced_X_test, y_test)
        torch.save(test_dataset, os.path.join(dir_path, "test_dataset.pt"))

        # Model
        model = MLPRegression(len(pcs), 8, 1, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_func = MSELoss()
        # loss_func = RidgeLoss()
        # loss_func = LassoLoss(alpha=0.001)

        for r in range(comm_rounds):
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * lr
            model.train()
            kfold = KFold(n_splits=5, shuffle=True)
            # K-fold cross validation
            for fold, (train_inds, val_inds) in enumerate(kfold.split(train_dataset)):
                print(f"K-Fold ({fold+1}/{kfold.n_splits})")
                train_sampler = SubsetRandomSampler(train_inds)
                val_sampler = SubsetRandomSampler(val_inds)

                train_loader = DataLoader(train_dataset, batch_size=10, sampler=train_sampler)
                val_loader = DataLoader(train_dataset, batch_size=10, sampler=val_sampler)
                # Training epochs
                for n in range(epoch_num):  # Training epochs
                    print(f"Epoch {n+1}/{epoch_num}")
                    # Mini batches
                    for i, (X, y) in enumerate(train_loader, 0):  # Mini-batches
                        optimizer.zero_grad()
                        loss = 0
                        # A mini batch
                        for j in range(len(X)):  # Calculate on a mini-batch
                            prediction = model(reduced_X_train[i])
                            loss += loss_func(prediction[0], y_train[i], model)
                        loss /= len(X)  # Mean loss to do back prop
                        loss.backward()
                        optimizer.step()  # Update grad and bias for each mini-batch
            print(f"Round {r+1} finished\n")
            torch.save(model, os.path.join(dir_path, f"centralised_model_{r+1}.pt"))

        print("Done training")
    print("Argument set complete")
