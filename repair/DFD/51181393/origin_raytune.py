from keras.models import Sequential
from keras.layers import Dense
from numpy import linspace
from sklearn.model_selection import train_test_split

import random, os, pickle, sys

from ray import tune
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB

import numpy as np
from sklearn.model_selection import train_test_split

from keras.optimizers import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--algo", nargs="?", type=str, const="random", default="random", help="algorithm"
)
parser.add_argument(
    "--repeat",
    nargs="?",
    type=int,
    const=1,
    default=1,
    help="the number of repeats of study",
)
parser.add_argument(
    "--limit", nargs="?", type=int, const=1, default=1, help="time limit"
)
parser.add_argument(
    "--top10",
    nargs="?",
    type=int,
    const=1,
    default=1,
    help="whether to use only top 10 hyperparams (1:True, 0:False)",
)
args = parser.parse_args()
print(args)

# seeded_random = random.Random(time.time())

N_ITER = -1
EPOCHS_MULTIPLIER = random.uniform(1.1, 1.3)
TIME_LIMIT = args.limit
NUM_TRAINING = 1
ORIG_TRAINING_TIME_PATH = "/home/hpo/data/DeepFD_orig_training_time.txt"
PNUM = "51181393"


def train(config):
    # Generate dummy data
    data = data = linspace(1, 2, 100).reshape(-1, 1)
    y = data * 5
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=42
    )
    X_test_used, X_test_holdout, y_test_used, y_test_holdout = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42
    )

    # Define the model
    model = Sequential()
    model.add(
        Dense(
            1,
            input_dim=1,
            activation=config["act1"],
            kernel_initializer=config["ki1"],
            bias_initializer=config["bi1"],
        )
    )

    if config["optimizer"] == "sgd":
        optimizer = SGD(lr=config["lr"])
    elif config["optimizer"] == "adam":
        optimizer = Adam(lr=config["lr"])
    elif config["optimizer"] == "rmsprop":
        optimizer = RMSprop(lr=config["lr"])
    elif config["optimizer"] == "adagrad":
        optimizer = Adagrad(lr=config["lr"])
    elif config["optimizer"] == "adadelta":
        optimizer = Adadelta(lr=config["lr"])
    elif config["optimizer"] == "adamax":
        optimizer = Adamax(lr=config["lr"])
    elif config["optimizer"] == "nadam":
        optimizer = Nadam(lr=config["lr"])

    model.compile(optimizer=optimizer, loss=config["loss"], metrics=["mae"])

    model.fit(
        X_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=0,
    )

    _, val_loss = model.evaluate(X_test_used, y_test_used)
    _, holdout_loss = model.evaluate(X_test_holdout, y_test_holdout)
    tune.report(mean_loss=val_loss, holdout_loss=holdout_loss)


def run_tune(algo, repeat):
    if algo is None:
        algo = "hebo"

    with open("/home/hpo/data/raytune_predefined_params.pkl", "rb") as f:
        _params = pickle.load(f)

    used_params = ["loss", "batch_size", "optimizer", "lr"]
    params = {}
    for p in used_params:
        if p not in _params:
            continue
        params[p] = _params[p]

    params["ki1"] = _params["kernel_initializer"]
    params["bi1"] = _params["kernel_initializer"]
    params["act1"] = _params["activation_function"]

    initial_params = {
        "ki1": "glorot_uniform",
        "bi1": "zeros",
        "act1": "linear",
        "unit1": 1,
        "loss": "mean_squared_error",
        "batch_size": 32,
        "optimizer": "rmsprop",
        "lr": 0.001,
        "epochs": 200,
    }

    initial_params = [initial_params]

    with open(ORIG_TRAINING_TIME_PATH, "r") as f:
        train_times = []
        for line in f:
            pnum, train_time = line.strip().split("|")
            if pnum == PNUM:
                train_times.append(float(train_time))

    # Set epochs
    epochs = [initial_params[0]["epochs"]]
    for x in range(5):
        val = int(initial_params[0]["epochs"] * (EPOCHS_MULTIPLIER**x))
        epochs.append(val)

        val = int(initial_params[0]["epochs"] / (EPOCHS_MULTIPLIER**x))
        if val > 1:
            epochs.append(val)
    epochs = tune.choice(list(set(epochs)))
    params["epochs"] = epochs

    # Set units
    units = ["unit1"]
    for unit in units:
        params[unit] = tune.choice(
            [
                int(initial_params[0][unit] / 2),
                initial_params[0][unit],
                initial_params[0][unit] * 2,
            ]
        )

    # Set layers
    layers = []
    for layer in layers:
        params[layer] = tune.choice([0, 1, 2])

    # Add initial parameter value to params if not in params
    for k, v in initial_params[0].items():
        if k == "lr" or "dropout" in k:
            continue
        if v not in params[k].categories:
            params[k] = tune.choice(params[k].categories + [v])

    if args.top10 == 1:
        top10 = [
            "loss",
            "batch_size",
            "optimizer",
            "lr",
            "epochs",
            "act1",
            "ki1",
            "bi1",
            "unit1",
        ]

        for k in list(params):
            if k not in top10:
                params[k] = tune.choice([initial_params[0][k]])

    if algo == "hebo":
        search_alg = HEBOSearch(
            metric="mean_loss", mode="min", points_to_evaluate=initial_params
        )
    elif algo == "random":
        search_alg = BasicVariantGenerator(points_to_evaluate=initial_params)
    elif algo == "bohb":
        bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=1)
        bohb_search = TuneBOHB(points_to_evaluate=initial_params)

    if algo in ["hebo", "random"]:
        analysis = tune.run(
            train,
            name=f"train_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}",
            metric="mean_loss",
            mode="min",
            time_budget_s=np.mean(train_times) * TIME_LIMIT,
            num_samples=N_ITER,
            resources_per_trial={"cpu": 40, "gpu": 1},
            config=params,
            verbose=1,
            search_alg=search_alg,
            local_dir="./",
            raise_on_failed_trial=False,
        )
    elif algo == "bohb":
        analysis = tune.run(
            train,
            name=f"train_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}",
            metric="mean_loss",
            mode="min",
            time_budget_s=np.mean(train_times) * TIME_LIMIT,
            num_samples=N_ITER,
            resources_per_trial={"cpu": 40, "gpu": 1},
            config=params,
            verbose=1,
            scheduler=bohb_scheduler,
            search_alg=bohb_search,
            local_dir="./",
            raise_on_failed_trial=False,
        )
    else:
        analysis = tune.run(
            train,
            name=f"train_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}",
            metric="mean_loss",
            mode="min",
            time_budget_s=np.mean(train_times) * TIME_LIMIT,
            num_samples=N_ITER,
            resources_per_trial={"cpu": 40, "gpu": 1},
            config=params,
            verbose=1,
            scheduler=search_alg,
            local_dir="./",
            raise_on_failed_trial=False,
        )

    print("Best hyperparameters found were: ", analysis.best_config)
    results_df = analysis.results_df
    results_df.to_csv(
        f"/home/hpo/results/DFD_|{PNUM}|_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}_repeat_{repeat}_top10_{args.top10}.csv"
    )


for repeat in range(args.repeat):
    run_tune(args.algo, repeat)
