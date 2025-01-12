# from Utils.utils import save_pkl, pack_train_config
from keras.layers import Dense, merge, Activation, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import random, os, pickle, sys
import numpy as np

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

N_ITER = -1
EPOCHS_MULTIPLIER = random.uniform(1.1, 1.3)
TIME_LIMIT = args.limit
NUM_TRAINING = 1
ORIG_TRAINING_TIME_PATH = "/home/hpo/data/DeepFD_orig_training_time.txt"
PNUM = "41600519"


def train(config):
    df = pd.read_csv("/home/hpo/DFD/41600519/training.csv")
    y = np.array(df["Label"].apply(lambda x: 0 if x == "s" else 1))
    X = np.array(df.drop(["EventId", "Label"], axis=1))

    X_train, X_test, y_train, y_test = train_test_split(
        X[:10000], y[:10000], test_size=0.3, random_state=42
    )
    X_test_used, X_test_holdout, y_test_used, y_test_holdout = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42
    )

    model = Sequential()
    """
    ValueError: Error when checking input: expected dense_1_input to have shape (30,) but got array with shape (31,)
    """
    model.add(
        Dense(
            600,
            input_shape=(31,),
            kernel_initializer=config["ki1"],
            bias_initializer=config["bi1"],
            activation=config["act1"],
        )
    )
    model.add(Dropout(0.5))
    for _ in range(config["layer2"]):
        model.add(
            Dense(
                config["unit2"],
                kernel_initializer=config["ki2"],
                bias_initializer=config["bi2"],
                activation=config["act2"],
            )
        )
    model.add(Dropout(0.5))
    for _ in range(config["layer3"]):
        model.add(
            Dense(
                config["unit3"],
                kernel_initializer=config["ki3"],
                bias_initializer=config["bi3"],
                activation=config["act3"],
            )
        )
    model.add(Dropout(0.5))
    model.add(
        Dense(
            1,
            kernel_initializer=config["ki4"],
            bias_initializer=config["bi4"],
            activation=config["act4"],
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

    model.compile(loss=config["loss"], optimizer=optimizer, metrics=["accuracy"])
    model.fit(
        X_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=0,
    )

    _, val_acc = model.evaluate(X_test_used, y_test_used)
    _, holdout_acc = model.evaluate(X_test_holdout, y_test_holdout)
    tune.report(mean_accuracy=val_acc, holdout_acc=holdout_acc)


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

    params["ki1"] = params["ki2"] = params["ki3"] = params["ki4"] = _params["kernel_initializer"]
    params["bi1"] = params["bi2"] = params["bi3"] = params["bi4"] = _params["kernel_initializer"]
    params["act1"] = params["act2"] = params["act3"] = params["act4"] = _params["activation_function"]

    initial_params = {
            "ki1": "glorot_uniform",
            "ki2": "glorot_uniform",
            "ki3": "glorot_uniform",
            "ki4": "glorot_uniform",
            "bi1": "zeros",
            "bi2": "zeros",
            "bi3": "zeros",
            "bi4": "zeros",
            "act1": "relu",
            "act2": "relu",
            "act3": "relu",
            "act4": "sigmoid",
            "layer2": 1,
            "layer3": 1,
            "unit2": 400,
            "unit3": 100,
            "loss": "binary_crossentropy",
            "batch_size": 1,
            "optimizer": "rmsprop",
            "lr": 0.001,
            "epochs": 1,
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
    units = ["unit2", "unit3"]
    for unit in units:
        params[unit] = tune.choice(
            [
                int(initial_params[0][unit] / 2),
                initial_params[0][unit],
                initial_params[0][unit] * 2,
            ]
        )

    # Set layers
    layers = ["layer2", "layer3"]
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
            "act2",
            "act3",
            "act4",
            "ki1",
            "ki2",
            "ki3",
            "ki4",
            "bi1",
            "bi2",
            "bi3",
            "bi4",
            "unit2",
            "unit3"
            "layer2",
            "layer3"
        ]

        for k in list(params):
            if k not in top10:
                params[k] = tune.choice([initial_params[0][k]])

    if algo == "hebo":
        search_alg = HEBOSearch(
            metric="mean_accuracy", mode="max", points_to_evaluate=initial_params
        )
    elif algo == "random":
        search_alg = BasicVariantGenerator(points_to_evaluate=initial_params)
    elif algo == "bohb":
        bohb_scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=1
        )
        bohb_search = TuneBOHB(points_to_evaluate=initial_params)

    if algo in ["hebo", "random"]:
        analysis = tune.run(
            train,
            name=f"train_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}",
            metric="mean_accuracy",
            mode="max",
            stop={"mean_accuracy": 0.99},
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
            metric="mean_accuracy",
            mode="max",
            stop={"mean_accuracy": 0.99},
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
            metric="mean_accuracy",
            mode="max",
            stop={"mean_accuracy": 0.99},
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
        f"/home/hpo/results/DFD/DFD_|{PNUM}|_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}_repeat_{repeat}_top10_{args.top10}.csv"
    )


for repeat in range(args.repeat):
    run_tune(args.algo, repeat)
