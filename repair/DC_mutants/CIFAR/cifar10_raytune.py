import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split

import random, os, pickle, sys, json
import ray

from ray import tune
from ray.tune.integration.keras import TuneReportCallback
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
    # Model configuration
    batch_size = 64
    img_width, img_height, img_num_channels = 32, 32, 3
    loss_function = sparse_categorical_crossentropy
    no_classes = 10
    no_epochs = 50
    optimizer = Adam()
    validation_split = 0.2
    verbosity = 1

    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Determine shape of the data
    input_shape = (img_width, img_height, img_num_channels)
    # Parse numbers as floats
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # Normalize data
    x_train = x_train / 255
    x_test = x_test / 255

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=validation_split, random_state=0
    )
    X_test_used, X_test_holdout, y_test_used, y_test_holdout = train_test_split(
        x_test, y_test, test_size=0.2, random_state=42
    )

    # Create the model
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
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            64,
            kernel_size=(3, 3),
            activation=config["act2"],
            kernel_initializer=config["ki2"],
            bias_initializer=config["bi2"],
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            128,
            kernel_size=(3, 3),
            activation=config["act3"],
            kernel_initializer=config["ki3"],
            bias_initializer=config["bi3"],
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    for _ in range(config["layer1"]):
        model.add(
            Dense(
                config["unit1"],
                activation=config["act4"],
                kernel_initializer=config["ki4"],
                bias_initializer=config["bi4"],
            )
        )
    for _ in range(config["layer2"]):
        model.add(
            Dense(
                config["unit2"],
                activation=config["act5"],
                kernel_initializer=config["ki5"],
                bias_initializer=config["bi5"],
            )
        )
    model.add(
        Dense(
            no_classes,
            activation=config["act6"],
            kernel_initializer=config["ki6"],
            bias_initializer=config["bi6"],
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

    # Compile the model
    model.compile(loss=config["loss"], optimizer=optimizer, metrics=["accuracy"])

    # Fit data to model
    history = model.fit(
        x_train,
        y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        verbose=0,
    )

    # Generate generalization metrics

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

    for i in range(1, 7):
        params[f"act{i}"] = _params["activation_function"]
        params[f"ki{i}"] = params[f"bi{i}"] = _params["kernel_initializer"]

    with open("./initial_params/cifar10_conv.json", "r") as f:
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
    units = ["unit1", "unit2"]
    for unit in units:
        params[unit] = tune.choice(
            [
                int(initial_params[0][unit] / 2),
                initial_params[0][unit],
                initial_params[0][unit] * 2,
            ]
        )

    # Set layers
    layers = ["layer1", "layer2"]
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
            "act5",
            "act6",
            "ki1",
            "ki2",
            "ki3",
            "ki4",
            "ki5",
            "ki6",
            "bi1",
            "bi2",
            "bi3",
            "bi4",
            "bi5",
            "bi6",
            "unit1",
            "unit2",
            "layer1",
            "layer2",
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
        f"/home/hpo/results/DC_CIFAR_|{mutant}|_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}_repeat_{repeat}_top10_{args.top10}.csv"
    )


for mutant in os.listdir("./"):
    if (
        ".py" not in mutant
        or mutant in ["cifar10_conv.py", "cifar10_raytune.py"]
        or not os.path.isfile(mutant)
    ):
        continue
    
    print("=" * 30)
    print(mutant)
    for repeat in range(args.repeat):
        run_tune(args.algo, mutant, repeat)
