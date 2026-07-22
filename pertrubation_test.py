import gc
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from custom_loss_functions import WeightedCRPS, create_bin_weights
from data_prep import PreparingData
from model_classes_test import ACORN

pd.options.mode.chained_assignment = None

working_dir = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Loading CONFIG json file
with open("prime_config.json", "r") as f:
    CONFIG = json.load(f)

if not os.path.exists(CONFIG["model_dir"]):
    os.makedirs(CONFIG["model_dir"])


class Early_Stopping:
    """
    Class to create an early stopping condition for the model.
    """

    def __init__(self, model_file, decreasing_loss_patience=25, model_config=CONFIG["model_config"]):
        # initializing the variables
        self.model_file = model_file  # dynamically tracking model path
        self.decreasing_loss_patience = decreasing_loss_patience
        self.loss_counter = 0
        self.training_counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.model_config = model_config

    def save_checkpoint(self, val_loss):
        # saving the model if the validation loss is less than the best loss
        self.best_loss = val_loss
        print("Saving checkpoint!")

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_epoch": self.best_epoch,
                "finished_training": False,
                "model_config": self.model_config,
            },
            self.model_file,
        )

    def __call__(self, train_loss, val_loss, model, optimizer, epoch):
        self.model = model
        self.optimizer = optimizer
        if self.best_score is None:
            self.best_train_loss = train_loss
            self.best_score = val_loss
            self.best_loss = val_loss
            self.save_checkpoint(val_loss)
            self.best_epoch = epoch

        elif val_loss >= self.best_score:
            self.loss_counter += 1
            if self.loss_counter >= self.decreasing_loss_patience:
                gc.collect()
                print(
                    f"Engaging Early Stopping due to lack of improvement in validation loss. Best model saved at epoch {self.best_epoch} with a training loss of {self.best_train_loss} and a validation loss of {self.best_score}"
                )
                return True
        else:
            self.best_train_loss = train_loss
            self.best_score = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(val_loss)
            self.loss_counter = 0
            self.training_counter = 0
            return False


def resume_training(model, optimizer, model_file):
    try:
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["best_epoch"]
        finished_training = checkpoint["finished_training"]
    except KeyError:
        model.load_state_dict(torch.load(model_file))
        optimizer = None
        epoch = 0
        finished_training = True

    return model, optimizer, epoch, finished_training


def fit_model(model, empty_model, train, val, model_file, val_loss_patience=25, num_epochs=500, bin_edges=None, bin_weights=None, model_config=None):
    bin_edges = bin_edges.to(DEVICE)
    bin_weights = bin_weights.to(DEVICE)
    criterion = WeightedCRPS(bin_edges=bin_edges, bin_weights=bin_weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    if os.path.exists(model_file):
        model, optimizer, current_epoch, finished_training = resume_training(model=model, optimizer=optimizer, model_file=model_file)
    else:
        finished_training = False
        current_epoch = 0

    if current_epoch is None:
        current_epoch = 0

    if not finished_training:
        train_loss_list, val_loss_list = [], []
        model.to(DEVICE)

        # passed model_file to early stopping
        early_stopping = Early_Stopping(model_file=model_file, decreasing_loss_patience=val_loss_patience, model_config=model_config)

        while current_epoch < num_epochs:
            stime = time.time()
            model.train()
            running_training_loss, running_val_loss = 0.0, 0.0

            for X, y in tqdm.tqdm(train):
                X = X.to(DEVICE, dtype=torch.float)
                y = y.to(DEVICE, dtype=torch.float)
                X = X.unsqueeze(1)

                output = model(X)
                output = output.squeeze()
                loss = criterion(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                X = X.to("cpu")
                y = y.to("cpu")
                running_training_loss += loss.to("cpu").item()

            model.eval()
            for X, y in tqdm.tqdm(val):
                X = X.to(DEVICE, dtype=torch.float)
                y = y.to(DEVICE, dtype=torch.float)
                X = X.unsqueeze(1)

                with torch.no_grad():
                    output = model(X)
                    output = output.squeeze()
                    val_loss = criterion(output, y)
                    X = X.to("cpu")
                    y = y.to("cpu")
                    running_val_loss += val_loss.to("cpu").item()

            loss = running_training_loss / len(train)
            val_loss = running_val_loss / len(val)

            train_loss_list.append(loss)
            val_loss_list.append(val_loss)

            if (early_stopping(train_loss=loss, val_loss=val_loss, model=model, optimizer=optimizer, epoch=current_epoch)) or (
                current_epoch == num_epochs - 1
            ):
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

                model = None
                model = empty_model
                final = torch.load(model_file)
                final["finished_training"] = True
                model.load_state_dict(final["model"])
                torch.save(final, model_file)
                break

            epoch_time = time.time() - stime
            print(f"Epoch [{current_epoch}/{num_epochs}], Loss: {loss:.4f} Validation Loss: {val_loss:.4f} Epoch Time: {epoch_time:.2f} seconds")
            torch.cuda.empty_cache()
            current_epoch += 1

        loss_tracker = pd.DataFrame({"train_loss": train_loss_list, "val_loss": val_loss_list})

        if not os.path.exists(working_dir + "loss_tracker"):
            os.makedirs(working_dir + "loss_tracker")

        # Ensure the loss tracker saves using the specific version
        version = model_file.split("/")[-1].split(".")[0]
        loss_tracker.to_feather(working_dir + f"loss_tracker/{version}_loss_tracker.feather")

        gc.collect()

    else:
        try:
            final = torch.load(model_file)
            model.load_state_dict(final["model"])
        except KeyError:
            model.load_state_dict(torch.load(model_file))

    return model


def evaluation(model, test, test_dict):
    output, xtest_list, ytest_list = [], [], []
    model.eval()
    running_loss = 0.0
    model.to(DEVICE, dtype=torch.float)

    with torch.no_grad():
        for x, y in tqdm.tqdm(test):
            x = x.to(DEVICE, dtype=torch.float)
            y = y.to(DEVICE, dtype=torch.float)
            x = x.unsqueeze(1)

            predicted = model(x)
            predicted = predicted.squeeze()
            predicted = predicted[:, :, :, 1:-1]
            loss = F.mse_loss(predicted[:, 0, :, :], y)
            running_loss += loss.item()

            if predicted.get_device() != -1:
                predicted = predicted.to("cpu")
            if x.get_device() != -1:
                x = x.to("cpu")
            if y.get_device() != -1:
                y = y.to("cpu")

            predicted = torch.squeeze(predicted, dim=1).numpy()
            output.append(predicted)
            x = torch.squeeze(x, dim=1).numpy()

    output = np.concatenate(output, axis=0)
    print(output.shape)
    print(f"Evaluation Loss: {running_loss / len(test)}")

    for pred, key in zip(output, test_dict.keys()):
        test_dict[key]["predicted"] = pred

    return test_dict


def main():
    if not os.path.exists(working_dir + "/outputs"):
        os.makedirs(working_dir + "/outputs")
    if not os.path.exists(working_dir + "/models"):
        os.makedirs(working_dir + "/models")

    # --- DEFINING EXPERIMENTS ---
    experiments = [
        {
            "experiment_name": "Model WITHOUT STDs",
            "version": "PRIME_1_NO_STD",
            "sw_data": "<PLACEHOLDER_FILE_WITHOUT_STD.feather>",
            "input_params": ["Vx", "BX_GSE", "BY_GSM", "BZ_GSM", "proton_density", "sin_month", "cos_month", "F107"],
        },
        {
            "experiment_name": "Model WITH STDs",
            "version": "PRIME_1_WITH_STD",
            "sw_data": "<PLACEHOLDER_FILE_WITH_STD.feather>",
            "input_params": [
                "Vx",
                "BX_GSE",
                "BY_GSM",
                "BZ_GSM",
                "proton_density",
                "sin_month",
                "cos_month",
                "F107",
                "<STD_PLACEHOLDER_1>",
                "<STD_PLACEHOLDER_2>",
            ],
        },
    ]

    for exp in experiments:
        print(f"\n{'=' * 50}")
        print(f"STARTING EXPERIMENT: {exp['experiment_name']}")
        print(f"{'=' * 50}\n")

        # Establishing dynamic paths based on current experiment
        current_version = exp["version"]
        model_file = f"{CONFIG['model_dir']}{CONFIG['model']}_{current_version}_{CONFIG['eras']}.pt"

        print("Loading data...")
        # Passing kwargs to PreparingData to override JSON config defaults
        PD = PreparingData(
            config="prime", sw_data=exp["sw_data"], input_params=exp["input_params"], version=current_version, data_version=current_version
        )
        train_dict, val_dict, test_dict = PD()

        train_x, train_y = [train_dict[key]["input"] for key in train_dict.keys()], [train_dict[key]["ampere"] for key in train_dict.keys()]
        val_x, val_y = [val_dict[key]["input"] for key in val_dict.keys()], [val_dict[key]["ampere"] for key in val_dict.keys()]
        test_x, test_y = [test_dict[key]["input"] for key in test_dict.keys()], [test_dict[key]["ampere"] for key in test_dict.keys()]

        bin_weights, hist, bin_edges = create_bin_weights(
            np.hstack(train_y),
            num_bins=[0, np.percentile(np.abs(np.hstack(train_y)), 95), np.max(np.abs(np.hstack(train_y)))],
            range_min=None,
            range_max=None,
        )
        bin_weights = torch.tensor(bin_weights)
        bin_edges = torch.tensor(bin_edges)

        train_y = np.array([np.concatenate((Y[:, -2:-1], Y, Y[:, 0:1]), axis=1) for Y in train_y])
        val_y = np.array([np.concatenate((Y[:, -2:-1], Y, Y[:, 0:1]), axis=1) for Y in val_y])

        train = DataLoader(list(zip(train_x, train_y)), batch_size=CONFIG["batch_size"], shuffle=True)
        val = DataLoader(list(zip(val_x, val_y)), batch_size=CONFIG["batch_size"], shuffle=True)
        test = DataLoader(list(zip(test_x, test_y)), batch_size=CONFIG["batch_size"], shuffle=False)

        print("Creating model....")
        torch.manual_seed(CONFIG["random_seed"])
        torch.cuda.manual_seed(CONFIG["random_seed"])

        output_size = train_y[0].shape
        model_config = CONFIG["model_config"].copy()
        model_config["output_size"] = output_size

        # Important: dynamically adjust `in_channels` based on the length of input parameters
        # Currently, your input data shape relies on this parameter matching the metric list size
        model_config["in_channels"] = 1  # Update this if the model architecture expects input depth to match parameter counts

        model = ACORN(**model_config)
        model.to(DEVICE)

        print("Fitting model....")
        model = fit_model(
            model=model,
            empty_model=model,
            train=train,
            val=val,
            model_file=model_file,  # Now dynamically assigned
            val_loss_patience=CONFIG["early_stop_patience"] or 25,
            num_epochs=CONFIG["epochs"],
            bin_edges=bin_edges,
            bin_weights=bin_weights,
            model_config=model_config,
        )

        print("Making predictions....")
        test_dict = evaluation(model, test, test_dict)

        print("Saving results....")
        output_file_name = (
            working_dir + f"/outputs/{CONFIG['model']}_{current_version}_{CONFIG['eras']}_storm_training_{CONFIG['extract_storms']}_results.pkl"
        )
        with open(output_file_name, "wb") as f:
            pickle.dump(test_dict, f)

        gc.collect()


if __name__ == "__main__":
    main()
    print("It ran. Good job!")
