import pickle

# from Utils.utils import save_pkl, pack_train_config
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import keras

import random, os, pickle, sys, time

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

seeded_random = random.Random(time.time())

N_ITER = -1
EPOCHS_MULTIPLIER = seeded_random.uniform(1.1, 1.3)
TIME_LIMIT = args.limit
NUM_TRAINING = 1
ORIG_TRAINING_TIME_PATH = "/home/hpo/data/DeepFD_orig_training_time.txt"
PNUM = "56380303"


RADIX = 7
np.random.seed(42)


def train(config):
    def _get_number(vector):
        return sum(x * 2**i for i, x in enumerate(vector))

    def _get_mod_result(vector):
        return _get_number(vector) % RADIX

    def _number_to_vector(number):
        binary_string = bin(number)[2:]
        if len(binary_string) > 20:
            raise NotImplementedError
        bits = (((0,) * (20 - len(binary_string))) + tuple(map(int, binary_string)))[
            ::-1
        ]
        assert len(bits) == 20
        return np.c_[bits]

    def get_mod_result_vector(vector):
        return _number_to_vector(_get_mod_result(vector))

    X = np.random.randint(2, size=(10000, 20))
    Y = np.vstack(map(get_mod_result_vector, X))

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    X_test_used, X_test_holdout, y_test_used, y_test_holdout = train_test_split(
        x_test, y_test, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(
        Dense(
            units=config["unit1"],
            input_dim=20,
            activation=config["act1"],
            kernel_initializer=config["ki1"],
            bias_initializer=config["bi1"],
        )
    )
    for _ in range(config["layer2"]):
        model.add(
            Dense(
                units=config["unit2"],
                activation=config["act2"],
                kernel_initializer=config["ki2"],
                bias_initializer=config["bi2"],
            )
        )
    model.add(
        Dense(
            units=20,
            activation=config["act3"],
            kernel_initializer=config["ki3"],
            bias_initializer=config["bi3"],
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

    model.compile(optimizer=optimizer, loss=config["loss"], metrics=["accuracy"])

    model.fit(
        x_train,
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

    params["ki1"] = params["ki2"] = params["ki3"] = _params["kernel_initializer"]
    params["bi1"] = params["bi2"] = params["bi3"] = _params["kernel_initializer"]
    params["act1"] = params["act2"] = params["act3"] = _params["activation_function"]

    initial_params = {
        "ki1": "glorot_uniform",
        "ki2": "glorot_uniform",
        "ki3": "glorot_uniform",
        "bi1": "zeros",
        "bi2": "zeros",
        "bi3": "zeros",
        "act1": "relu",
        "act2": "relu",
        "act3": "softmax",
        "layer2": 1,
        "unit1": 20,
        "unit2": 20,
        "loss": "categorical_crossentropy",
        "batch_size": 50,
        "optimizer": "sgd",
        "lr": 0.01,
        "epochs": 10,
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
    layers = ["layer2"]
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
            "ki1",
            "ki2",
            "ki3",
            "bi1",
            "bi2",
            "bi3",
            "unit1",
            "unit2",
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
    elif algo == "bohb":
        bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=1)
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
        f"/home/hpo/results/DFD_|{PNUM}|_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}_repeat_{repeat}_top10_{args.top10}.csv"
    )


for repeat in range(args.repeat):
    run_tune(args.algo, repeat)
