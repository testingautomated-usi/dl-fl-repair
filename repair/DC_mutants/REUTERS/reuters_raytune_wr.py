import keras
import os
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
import numpy as np

import random, os, pickle, sys, json, time
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

seeded_random = random.Random(time.time())

N_ITER = -1
EPOCHS_MULTIPLIER = seeded_random.uniform(1.1, 1.3)
TIME_LIMIT = args.limit
NUM_TRAINING = 1
ORIG_TRAINING_TIME_PATH = "/home/hpo/data/DC_orig_training_time.txt"

random.seed(42)
np.random.seed(42)


def train(config):
    (x_train_init, y_train_init), (x_test, y_test) = reuters.load_data(
        num_words=None, test_split=0.2, seed=42
    )
    word_index = reuters.get_word_index(path="reuters_word_index.json")
    num_classes = max(y_train_init) + 1
    index_to_word = {}
    for key, value in word_index.items():
        index_to_word[value] = key

    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    x_train_init = tokenizer.sequences_to_matrix(x_train_init, mode="binary")
    x_test = tokenizer.sequences_to_matrix(x_test, mode="binary")

    y_train_init = keras.utils.to_categorical(y_train_init, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_init, y_train_init, test_size=0.2, random_state=42
    )
    X_test_used, X_test_holdout, y_test_used, y_test_holdout = train_test_split(
        x_test, y_test, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(
        Dense(
            config["unit1"],
            input_shape=(max_words,),
            activation=config["act1"],
            kernel_initializer=config["ki1"],
            bias_initializer=config["bi1"],
            kernel_regularizer='l1'
        )
    )
    model.add(Dropout(0.5))
    model.add(
        Dense(
            num_classes,
            activation=config["act2"],
            kernel_initializer=config["ki2"],
            bias_initializer=config["bi2"],
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

    params["ki1"] = params["ki2"] = _params["kernel_initializer"]
    params["bi1"] = params["bi2"] = _params["kernel_initializer"]
    params["act1"] = params["act2"] = _params["activation_function"]

    with open("./initial_params/reuters.json", "r") as f:
        initial_params = json.load(f)
    # with open(f"./initial_params/mutant_params.json", "r") as f:
    #     mutant_params = json.load(f)
    #     mutant_params = mutant_params[mutant]

    # for k, v in mutant_params.items():
    #     if k in initial_params:
    #         initial_params[k] = v
    #     else:
    #         pass  # TODO

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
            "act2",
            "ki1",
            "ki2",
            "bi1",
            "bi2",
            "unit1"
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
        f"/home/hpo/results/DC_REUTERS_|{mutant}|_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}_repeat_{repeat}_top10_{args.top10}.csv"
    )


for mutant in os.listdir("./"):
    if (
        ".py" not in mutant
        or mutant in ["reuters.py", "reuters_raytune.py"]
        or not os.path.isfile(mutant)
    ):
        continue
    
    if mutant != "reuters_add_weights_regularisation_mutated0_MP_l1_0.py":
        continue
    
    print("=" * 30)
    print(mutant)
    for repeat in range(args.repeat):
        run_tune(args.algo, mutant, repeat)