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
from Decorator import *


class Centralised:

    def __init__(self):
        self.client_id = "Centralised"
        self.checkpoint_dir = "./centralised_checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    # @hyper_tune
    def train_epochs(self, train_loader, val_loader, optimizer, model, loss_func):
        # Training epochs
        for n in range(self.epoch_num):  # Training epochs
            print(f"    Epoch {n + 1}/{self.epoch_num}")
            # Mini batches
            for i, (X, y) in enumerate(train_loader, 0):  # Mini-batches
                optimizer.zero_grad()
                loss = 0
                # A mini batch
                for j in range(len(X)):  # Calculate on a mini-batch
                    prediction = model(X[j])
                    loss += loss_func(prediction[0], y[j], model)
                loss /= len(X)  # Mean loss to do back prop
                loss.backward()
                optimizer.step()  # Update grad and bias for each mini-batch
        with torch.no_grad():
            cv_loss_func = MSELoss()  # Use a separate loss function for cross validation
            cv_loss = 0
            for j, (features, target) in enumerate(val_loader, 0):
                prediction = model(features)
                cv_loss += float(cv_loss_func(prediction[0], target, model))
        print(f"    lr: {optimizer.param_groups[0]['lr']}; "
              f"momentum: {optimizer.param_groups[0]['momentum']}; "
              f"cv: {cv_loss}")
        return model, optimizer, cv_loss

    def work(self):
        # Torch setting
        torch.set_default_dtype(torch.float64)

        # Parameters
        data_path = "/Users/jacobvons/Downloads/Y4S1/COMP4540/federated_learning/all.csv"
        reader = ArgReader("./test_args.csv")
        reader.parse()
        for args in reader.args:
            self.epoch_num = int(args["e"])
            self.comm_rounds = int(args["rounds"])
            self.name = args["name"]
            self.explain_ratio = float(args["ratio"])

            self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.name)

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
                if sum(pca.explained_variance_ratio_) >= self.explain_ratio:
                    pc_num = len(pca.components_)
                    break
                else:
                    pca = PCA(n_components=pca.n_components + 1)
            print("At least", pc_num, "PCs")

            # Dimension Reduction
            if not os.path.exists("./central"):
                os.mkdir("./central")
            if not os.path.exists(f"./central/{self.name}"):
                os.mkdir(f"./central/{self.name}")
            dir_path = f"./central/{self.name}"
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
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            loss_func = MSELoss()
            # loss_func = RidgeLoss(alpha=0.001)
            # loss_func = LassoLoss(alpha=0.001)

            for r in range(self.comm_rounds):
                # for g in optimizer.param_groups:
                #     g["lr"] = g["lr"] * self.lr
                model.train()
                kfold = KFold(n_splits=5, shuffle=True)
                # K-fold cross validation
                max_cv = float("inf")
                best_model = None
                for fold, (train_inds, val_inds) in enumerate(kfold.split(train_dataset)):
                    print(f"K-Fold ({fold + 1}/{kfold.n_splits})")
                    train_sampler = SubsetRandomSampler(train_inds)
                    val_sampler = SubsetRandomSampler(val_inds)

                    train_loader = DataLoader(train_dataset, batch_size=10, sampler=train_sampler)
                    val_loader = DataLoader(train_dataset, batch_size=10, sampler=val_sampler)
                    model, optimizer, cv_loss = self.train_epochs(train_loader=train_loader,
                                                                  val_loader=val_loader, optimizer=optimizer,
                                                                  model=model, loss_func=loss_func)
                    print(f"Fold {fold+1} best hps: lr: {optimizer.param_groups[0]['lr']}, "
                          f"momentum: {optimizer.param_groups[0]['momentum']}, "
                          f"cv: {cv_loss}")
                    # Choose the best model according to cross validation score
                    if cv_loss < max_cv:
                        max_cv = cv_loss
                        best_model = model
                        print("Updated best model")
                print(f"Round {r + 1} finished\n")
                print(f"Best cv score: {max_cv}. Saving best round {r+1} model..")
                torch.save(best_model, os.path.join(dir_path, f"best_round_{(r+1):03}_model.pt"))

            print("Done training")
        print("Argument set complete")


if __name__ == "__main__":
    cent = Centralised()
    cent.work()
