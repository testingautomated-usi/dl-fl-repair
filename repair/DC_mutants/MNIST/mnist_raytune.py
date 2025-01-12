from __future__ import print_function
import keras, sys
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

import random, os, pickle, sys, json
import ray

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
ORIG_TRAINING_TIME_PATH = "/home/hpo/data/DC_orig_training_time.txt"


def train(config):
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    (img_rows, img_cols) = (28, 28)
    num_classes = 10

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    X_test_used, X_test_holdout, y_test_used, y_test_holdout = train_test_split(
        x_test, y_test, test_size=0.2, random_state=42
    )

    batch_size = 128
    epochs = 12
    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation=config["act1"],
            kernel_initializer=config["ki1"],
            bias_initializer=config["bi1"],
            input_shape=input_shape,
        )
    )
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation=config["act2"],
            kernel_initializer=config["ki2"],
            bias_initializer=config["bi2"],
        ),
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config["dropout1"]))
    model.add(Flatten())

    for _ in range(config["layer1"]):
        model.add(
            Dense(
                config["unit1"],
                activation=config["act3"],
                kernel_initializer=config["ki3"],
                bias_initializer=config["bi3"],
            )
        )
    model.add(Dropout(config["dropout2"]))
    model.add(
        Dense(
            num_classes,
            activation=config["act4"],
            kernel_initializer=config["ki4"],
            bias_initializer=config["bi4"],
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
        x_train,
        y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        verbose=0,
    )

    _, val_acc = model.evaluate(X_test_used, y_test_used)
    _, holdout_acc = model.evaluate(X_test_holdout, y_test_holdout)
    tune.report(mean_accuracy=val_acc, holdout_acc=holdout_acc)


def run_tune(algo, mutant, repeat):
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

    params["act1"] = params["act2"] = params["act3"] = params["act4"] = _params[
        "activation_function"
    ]
    params["ki1"] = params["ki2"] = params["ki3"] = params["ki4"] = _params[
        "kernel_initializer"
    ]
    params["bi1"] = params["bi2"] = params["bi3"] = params["bi4"] = _params[
        "kernel_initializer"
    ]
    params["dropout1"] = params["dropout2"] = _params["dropout"]

    with open("./initial_params/mnist_conv.json", "r") as f:
        initial_params = json.load(f)
    with open(f"./initial_params/mutant_params.json", "r") as f:
        mutant_params = json.load(f)
        mutant_params = mutant_params[mutant]

    for k, v in mutant_params.items():
        if k in initial_params:
            initial_params[k] = v
        else:
            pass  # TODO

    initial_params = [initial_params]

    with open(ORIG_TRAINING_TIME_PATH, "r") as f:
        train_times = []
        for line in f:
            pnum, train_time = line.strip().split("|")
            if pnum == mutant:
                train_times.append(float(train_time))

    # Set epochs
    epochs = [initial_params[0]["epochs"]]
    for x in range(5):
        val = int(initial_params[0]["epochs"] * (EPOCHS_MULTIPLIER**x))
        if val < 10000:
            epochs.append(val)

        val = int(initial_params[0]["epochs"] / (EPOCHS_MULTIPLIER**x))
        if val > 5:
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
    layers = ["layer1"]
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
            "unit1",
            "layer1",
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
    elif algo == "asha":
        search_alg = ASHAScheduler(
            time_attr="time_total_s",
            max_t=np.mean(train_times) * (EPOCHS_MULTIPLIER**4),
            grace_period=np.mean(train_times) / (EPOCHS_MULTIPLIER**4),
        )
    elif algo == "bohb":
        bohb_scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=1
        )
        bohb_search = TuneBOHB(points_to_evaluate=initial_params)

    if algo in ["hebo", "random"]:
        analysis = tune.run(
            train,
            name=f"{mutant}_train_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}",
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
            name=f"{mutant}_train_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}",
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
            name=f"{mutant}_train_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}",
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
        f"/home/hpo/results/DC_MNIST_|{mutant}|_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}_repeat_{repeat}_top10_{args.top10}.csv"
    )


for mutant in os.listdir("./"):
    if (
        ".py" not in mutant
        or mutant in ["mnist_conv.py", "mnist_raytune.py"]
        or not os.path.isfile(mutant)
    ):
        continue

    print("=" * 30)
    print(mutant)
    for repeat in range(args.repeat):
        run_tune(args.algo, mutant, repeat)
